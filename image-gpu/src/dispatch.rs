/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This file is part of paged (https://paged.media) and is additionally
 * available under the Paged Media Enterprise License (PMEL). Full
 * copyright and license information is available in LICENSE.md which is
 * distributed with this source code.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    MPL-2.0 OR Paged Media Enterprise License (PMEL)
 */

//! Batched dispatch (spec §9.2) — all invalid tiles of ONE node coalesce
//! into a single submit. For one `KernelPipeline` + one param block, N
//! tiles' bind groups are recorded into ONE command encoder and ONE
//! compute pass (one `dispatch_workgroups` per tile — the win is the
//! single submit, not fewer dispatches), then read back together.
//!
//! Each tile carries its own input/output/mask textures and bind groups,
//! exactly as `execute_tile_once` builds them, so the per-tile result is
//! identical to the single-tile path; only the encoder/pass/submit are
//! shared. That equivalence is the conformance gate
//! (`image-conformance/tests/dispatch_batch.rs`).

use image_kernels::abi;

use crate::{GpuContext, GpuError, KernelPipeline, TileInput};

const BYTES_PER_PIXEL: u32 = 8; // rgba16float

/// One tile's worth of work in a batch: input texel bytes (arity per the
/// pipeline's `KernelDef.inputs`), an optional selection mask (r16float;
/// `None` = the constant-1 Engine A binding), and the tile dimensions.
pub struct BatchTile<'a> {
    pub inputs: &'a [TileInput<'a>],
    pub mask: Option<&'a [u8]>,
    pub w: u32,
    pub h: u32,
}

/// Per-tile GPU resources, kept alive for the lifetime of the encoder so
/// the recorded bind groups stay valid until submit.
struct TileResources {
    out_tex: wgpu::Texture,
    readback: wgpu::Buffer,
    w: u32,
    h: u32,
    padded_row: u32,
    bind_groups: [wgpu::BindGroup; 4],
}

/// A batched dispatch over one pipeline + one param block. Build it, push
/// tiles, then `submit_and_read` to coalesce every tile into one submit.
pub struct DispatchBatch<'p> {
    pipeline: &'p KernelPipeline,
    params: Vec<u8>,
}

fn make_texture(
    ctx: &GpuContext,
    label: &str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn upload_f16(ctx: &GpuContext, tex: &wgpu::Texture, w: u32, h: u32, bytes: &[u8]) {
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * BYTES_PER_PIXEL),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

impl<'p> DispatchBatch<'p> {
    /// A batch over `pipeline` with one shared param block (the param
    /// bytes are `Params::as_bytes()`, validated against the layout).
    pub fn new(pipeline: &'p KernelPipeline, params: &[u8]) -> Result<Self, GpuError> {
        if params.len() != pipeline.def.params.size {
            return Err(GpuError::Kernel {
                kernel: pipeline.def.id,
                detail: format!(
                    "param block {} bytes, layout says {}",
                    params.len(),
                    pipeline.def.params.size
                ),
            });
        }
        Ok(DispatchBatch {
            pipeline,
            params: params.to_vec(),
        })
    }

    /// Build every tile's textures + bind groups, upload inputs, then
    /// record all dispatches into ONE encoder / ONE compute pass and one
    /// submit; reads every tile's output back. Returns one tightly-packed
    /// rgba16float `Vec<u8>` per input tile, in push order — byte-for-byte
    /// the same as calling `execute_tile_once` per tile.
    pub fn submit_and_read(
        &self,
        ctx: &GpuContext,
        tiles: &[BatchTile<'_>],
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        let def = self.pipeline.def;
        let one_f16 = 0x3C00u16.to_le_bytes();

        // Params uniform (shared across the batch).
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} batch params", def.id)),
            size: def.params.size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params_buf, 0, &self.params);

        let in_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let mut resources: Vec<TileResources> = Vec::with_capacity(tiles.len());

        for (ti, tile) in tiles.iter().enumerate() {
            if tile.inputs.len() != def.inputs as usize {
                return Err(GpuError::Kernel {
                    kernel: def.id,
                    detail: format!(
                        "tile {ti}: expected {} inputs, got {}",
                        def.inputs,
                        tile.inputs.len()
                    ),
                });
            }
            let (w, h) = (tile.w, tile.h);

            // Inputs.
            let in_textures: Vec<wgpu::Texture> = tile
                .inputs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let tex = make_texture(
                        ctx,
                        &format!("{} t{ti} in{i}", def.id),
                        w,
                        h,
                        wgpu::TextureFormat::Rgba16Float,
                        in_usage,
                    );
                    upload_f16(ctx, &tex, w, h, t.f16_bytes);
                    tex
                })
                .collect();
            let in_views: Vec<wgpu::TextureView> = in_textures
                .iter()
                .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
                .collect();

            // Selection mask (constant-1 default — the Engine A binding).
            let constant_one: Vec<u8>;
            let mask_bytes: &[u8] = match tile.mask {
                Some(m) => m,
                None => {
                    constant_one = one_f16
                        .iter()
                        .copied()
                        .cycle()
                        .take((w * h * 2) as usize)
                        .collect();
                    &constant_one
                }
            };
            let mask_tex = make_texture(
                ctx,
                &format!("{} t{ti} mask", def.id),
                w,
                h,
                wgpu::TextureFormat::R16Float,
                in_usage,
            );
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &mask_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                mask_bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 2),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            let mask_view = mask_tex.create_view(&wgpu::TextureViewDescriptor::default());

            // Output storage texture.
            let out_tex = make_texture(
                ctx,
                &format!("{} t{ti} out", def.id),
                w,
                h,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            );
            let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

            // Bind groups (the frozen ABI, §9.2).
            let g0_entries: Vec<wgpu::BindGroupEntry> = in_views
                .iter()
                .enumerate()
                .map(|(i, v)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: wgpu::BindingResource::TextureView(v),
                })
                .collect();
            let g0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batch g0"),
                layout: &self.pipeline.group0,
                entries: &g0_entries,
            });
            let g1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batch g1"),
                layout: &self.pipeline.group1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                }],
            });
            let g2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batch g2"),
                layout: &self.pipeline.group2,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                }],
            });
            let g3 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batch g3"),
                layout: &self.pipeline.group3,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                }],
            });

            let row_bytes = w * BYTES_PER_PIXEL;
            let padded_row = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} t{ti} readback", def.id)),
                size: (padded_row as u64) * (h as u64),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            resources.push(TileResources {
                out_tex,
                readback,
                w,
                h,
                padded_row,
                bind_groups: [g0, g1, g2, g3],
            });
        }

        // ONE encoder, ONE compute pass — the coalesced dispatch (§9.2).
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{} batch", def.id)),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} batch", def.id)),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            for res in &resources {
                pass.set_bind_group(0, &res.bind_groups[0], &[]);
                pass.set_bind_group(1, &res.bind_groups[1], &[]);
                pass.set_bind_group(2, &res.bind_groups[2], &[]);
                pass.set_bind_group(3, &res.bind_groups[3], &[]);
                pass.dispatch_workgroups(
                    res.w.div_ceil(abi::WORKGROUP_SIZE),
                    res.h.div_ceil(abi::WORKGROUP_SIZE),
                    1,
                );
            }
        }
        for res in &resources {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &res.out_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &res.readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(res.padded_row),
                        rows_per_image: Some(res.h),
                    },
                },
                wgpu::Extent3d {
                    width: res.w,
                    height: res.h,
                    depth_or_array_layers: 1,
                },
            );
        }
        ctx.queue.submit([encoder.finish()]);

        // Map every readback, then poll once for the whole batch.
        let (tx, rx) = std::sync::mpsc::channel();
        for (i, res) in resources.iter().enumerate() {
            let tx = tx.clone();
            res.readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send((i, r));
                });
        }
        drop(tx);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        for _ in 0..resources.len() {
            let (_, r) = rx
                .recv()
                .map_err(|_| GpuError::Readback("map callback dropped".into()))?;
            r.map_err(|e| GpuError::Readback(format!("map_async: {e:?}")))?;
        }

        // Repack each tile to the tight (unpadded) rgba16float layout, in
        // push order.
        let mut outputs = Vec::with_capacity(resources.len());
        for res in &resources {
            let row_bytes = (res.w * BYTES_PER_PIXEL) as usize;
            let slice = res.readback.slice(..);
            let mut out = Vec::with_capacity(row_bytes * res.h as usize);
            {
                let data = slice.get_mapped_range();
                for row in 0..res.h {
                    let start = (row * res.padded_row) as usize;
                    out.extend_from_slice(&data[start..start + row_bytes]);
                }
            }
            res.readback.unmap();
            outputs.push(out);
        }
        Ok(outputs)
    }
}
