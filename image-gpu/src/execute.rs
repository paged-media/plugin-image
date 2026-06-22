/*
 * This file is part of paged (https://paged.media).
 *
 * paged is free software: you may redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License, version 3, as published by
 * the Free Software Foundation, OR under the Paged Media Enterprise License
 * (PMEL), a commercial license available from And The Next GmbH. Full
 * copyright and license information is available in LICENSE.md, distributed
 * with this source code.
 *
 * paged is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the licenses for details.
 *
 *  @copyright  Copyright (c) And The Next GmbH
 *  @license    AGPL-3.0-only OR Paged Media Enterprise License (PMEL)
 */

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

//! Single-tile kernel execution — the seed of the batched dispatcher
//! (M0 fan-out) and the path the conformance parity harness drives:
//! upload rgba16float input(s) + params + mask, one dispatch, read the
//! output back. Production batching coalesces all invalid tiles of a
//! node into one dispatch per pass (§9.2); this function is
//! deliberately the simplest correct realization of the same ABI.
//!
//! Two readback lanes over ONE recording path: the synchronous map
//! (native test path — `device.poll(wait)` blocks until the map
//! callback fired) and the ASYNC map ([`execute_tile_once_async`]) the
//! wasm32/WebGPU consumer needs — on the web `poll` cannot block (the
//! browser event loop delivers the callback), so the only correct
//! readback is to suspend until it does. Both lanes read the identical
//! recorded dispatch, so async output is byte-for-byte the sync output.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use image_kernels::{abi, KernelDef};

use crate::{GpuContext, GpuError, KernelPipeline};

/// One input tile: rgba16float texel bytes (8 bytes/px, tightly packed
/// rows).
pub struct TileInput<'a> {
    pub f16_bytes: &'a [u8],
}

const BYTES_PER_PIXEL: u32 = 8; // rgba16float

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

/// Execute a `module: true` kernel whose input window differs from the
/// output region (ABI v1.1: `Windowed` — input = out + 2·radius;
/// `Resample` — input = the source window). One input only (T1
/// conv/resample are unary). `win_bytes` is the rgba16float window at
/// `win_w`×`win_h`; mask + output are `out_w`×`out_h`.
#[allow(clippy::too_many_arguments)]
pub fn execute_windowed_once(
    ctx: &GpuContext,
    def: &'static KernelDef,
    win_bytes: &[u8],
    win_w: u32,
    win_h: u32,
    params: &[u8],
    mask: Option<&[u8]>,
    out_w: u32,
    out_h: u32,
) -> Result<Vec<u8>, GpuError> {
    if def.inputs != 1 {
        return Err(GpuError::Kernel {
            kernel: def.id,
            detail: "windowed execution is unary (T1)".into(),
        });
    }
    if params.len() != def.params.size {
        return Err(GpuError::Kernel {
            kernel: def.id,
            detail: format!(
                "param block {} bytes, layout says {}",
                params.len(),
                def.params.size
            ),
        });
    }
    let pipeline = KernelPipeline::build(ctx, def);
    let in_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    let in_tex = make_texture(
        ctx,
        &format!("{} window", def.id),
        win_w,
        win_h,
        wgpu::TextureFormat::Rgba16Float,
        in_usage,
    );
    upload_f16(ctx, &in_tex, win_w, win_h, win_bytes);
    let in_view = in_tex.create_view(&wgpu::TextureViewDescriptor::default());
    run_common(ctx, &pipeline, &[&in_view], def, params, mask, out_w, out_h)
}

/// Execute `def` over one `w`×`h` tile. `inputs.len()` must equal
/// `def.inputs`; `params` must be the param block's bytes
/// (`Params::as_bytes()`); `mask` is r16float texel bytes or `None`
/// for the constant-1 mask (the Engine A binding, §6.1). Returns the
/// output rgba16float bytes, tightly packed.
pub fn execute_tile_once(
    ctx: &GpuContext,
    def: &'static KernelDef,
    inputs: &[TileInput<'_>],
    params: &[u8],
    mask: Option<&[u8]>,
    w: u32,
    h: u32,
) -> Result<Vec<u8>, GpuError> {
    let (pipeline, in_views) = prepare_tile(ctx, def, inputs, params, w, h)?;
    run_common(ctx, &pipeline, &in_views, def, params, mask, w, h)
}

/// [`execute_tile_once`]'s ASYNC twin — the identical recorded dispatch
/// with an awaited readback map, for the wasm32/WebGPU realm where a
/// blocking `device.poll` cannot pump the map callback. Byte-for-byte
/// the sync output (asserted by the conformance async-parity test).
pub async fn execute_tile_once_async(
    ctx: &GpuContext,
    def: &'static KernelDef,
    inputs: &[TileInput<'_>],
    params: &[u8],
    mask: Option<&[u8]>,
    w: u32,
    h: u32,
) -> Result<Vec<u8>, GpuError> {
    let (pipeline, in_views) = prepare_tile(ctx, def, inputs, params, w, h)?;
    record_common(ctx, &pipeline, &in_views, def, params, mask, w, h)?
        .finish_async(ctx)
        .await
}

/// Shared head of the single-tile lanes: validate arity + param block,
/// build the pipeline, upload the rgba16float inputs.
fn prepare_tile(
    ctx: &GpuContext,
    def: &'static KernelDef,
    inputs: &[TileInput<'_>],
    params: &[u8],
    w: u32,
    h: u32,
) -> Result<(KernelPipeline, Vec<wgpu::TextureView>), GpuError> {
    if inputs.len() != def.inputs as usize {
        return Err(GpuError::Kernel {
            kernel: def.id,
            detail: format!("expected {} inputs, got {}", def.inputs, inputs.len()),
        });
    }
    if params.len() != def.params.size {
        return Err(GpuError::Kernel {
            kernel: def.id,
            detail: format!(
                "param block {} bytes, layout says {}",
                params.len(),
                def.params.size
            ),
        });
    }

    let pipeline = KernelPipeline::build(ctx, def);

    // Inputs (rgba16float, sampled via textureLoad).
    let in_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    let in_textures: Vec<wgpu::Texture> = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let tex = make_texture(
                ctx,
                &format!("{} in{i}", def.id),
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

    Ok((pipeline, in_views))
}

/// The shared dispatch tail: mask + params + output + bind groups +
/// one dispatch sized by the OUTPUT dims + synchronous readback. Input
/// views may be any size (point kernels: == output; windowed/resample:
/// the source window, ABI v1.1).
fn run_common(
    ctx: &GpuContext,
    pipeline: &KernelPipeline,
    in_views: &[impl std::borrow::Borrow<wgpu::TextureView>],
    def: &'static KernelDef,
    params: &[u8],
    mask: Option<&[u8]>,
    w: u32,
    h: u32,
) -> Result<Vec<u8>, GpuError> {
    record_common(ctx, pipeline, in_views, def, params, mask, w, h)?.finish_sync(ctx)
}

/// A submitted dispatch whose output awaits readback: the mapped-copy
/// buffer plus the row geometry to strip the COPY_BYTES_PER_ROW padding.
struct PendingReadback {
    readback: wgpu::Buffer,
    row_bytes: u32,
    padded_row: u32,
    h: u32,
}

impl PendingReadback {
    /// Native lane: a blocking `poll(wait)` pumps the map callback.
    fn finish_sync(self, ctx: &GpuContext) -> Result<Vec<u8>, GpuError> {
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| GpuError::Readback("map callback dropped".into()))?
            .map_err(|e| GpuError::Readback(format!("map_async: {e:?}")))?;
        Ok(self.collect())
    }

    /// Async lane: suspend until the map callback fires. On native the
    /// blocking poll still pumps it (so the await is immediately ready —
    /// the lane is natively testable under pollster); on wasm32 the
    /// browser event loop delivers it and the future suspends properly
    /// (`poll(wait)` cannot block on the web).
    async fn finish_async(self, ctx: &GpuContext) -> Result<Vec<u8>, GpuError> {
        let slice = self.readback.slice(..);
        let shared: Arc<Mutex<MapShared>> = Arc::default();
        let cb = shared.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let mut s = cb.lock().expect("map waker lock");
            s.result = Some(result);
            if let Some(w) = s.waker.take() {
                w.wake();
            }
        });
        #[cfg(not(target_arch = "wasm32"))]
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
        MapFuture(shared)
            .await
            .map_err(|e| GpuError::Readback(format!("map_async: {e:?}")))?;
        Ok(self.collect())
    }

    /// Strip the alignment padding out of the mapped rows.
    fn collect(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity((self.row_bytes * self.h) as usize);
        {
            let data = self.readback.slice(..).get_mapped_range();
            for row in 0..self.h {
                let start = (row * self.padded_row) as usize;
                out.extend_from_slice(&data[start..start + self.row_bytes as usize]);
            }
        }
        self.readback.unmap();
        out
    }
}

/// The map-callback rendezvous behind [`MapFuture`]. `Arc<Mutex<..>>`
/// because wgpu's native callback must be `Send`; the future itself
/// stays single-threaded.
#[derive(Default)]
struct MapShared {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

/// A tiny hand-rolled oneshot over the `map_async` callback — no
/// executor/futures dependency for one await point.
struct MapFuture(Arc<Mutex<MapShared>>);

impl Future for MapFuture {
    type Output = Result<(), wgpu::BufferAsyncError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut s = self.0.lock().expect("map waker lock");
        match s.result.take() {
            Some(r) => Poll::Ready(r),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// Record + submit the dispatch (everything up to, not including, the
/// readback map) — the ONE recording path both readback lanes share.
fn record_common(
    ctx: &GpuContext,
    pipeline: &KernelPipeline,
    in_views: &[impl std::borrow::Borrow<wgpu::TextureView>],
    def: &'static KernelDef,
    params: &[u8],
    mask: Option<&[u8]>,
    w: u32,
    h: u32,
) -> Result<PendingReadback, GpuError> {
    let in_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    // Selection mask (r16float; constant-1 default).
    let one_f16 = 0x3C00u16.to_le_bytes();
    let constant_one: Vec<u8>;
    let mask_bytes: &[u8] = match mask {
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
        &format!("{} mask", def.id),
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

    // Params uniform.
    let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{} params", def.id)),
        size: def.params.size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&params_buf, 0, params);

    // Output (storage, write-only — the portable path, §9.2).
    let out_tex = make_texture(
        ctx,
        &format!("{} out", def.id),
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
            resource: wgpu::BindingResource::TextureView(v.borrow()),
        })
        .collect();
    let g0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("g0"),
        layout: &pipeline.group0,
        entries: &g0_entries,
    });
    let g1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("g1"),
        layout: &pipeline.group1,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buf.as_entire_binding(),
        }],
    });
    let g2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("g2"),
        layout: &pipeline.group2,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&mask_view),
        }],
    });
    let g3 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("g3"),
        layout: &pipeline.group3,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&out_view),
        }],
    });

    // Readback buffer (row stride aligned to COPY_BYTES_PER_ROW_ALIGNMENT).
    let row_bytes = w * BYTES_PER_PIXEL;
    let padded_row =
        row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kernel readback"),
        size: (padded_row as u64) * (h as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(def.id),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(def.id),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &g0, &[]);
        pass.set_bind_group(1, &g1, &[]);
        pass.set_bind_group(2, &g2, &[]);
        pass.set_bind_group(3, &g3, &[]);
        pass.dispatch_workgroups(
            w.div_ceil(abi::WORKGROUP_SIZE),
            h.div_ceil(abi::WORKGROUP_SIZE),
            1,
        );
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);

    Ok(PendingReadback {
        readback,
        row_bytes,
        padded_row,
        h,
    })
}
