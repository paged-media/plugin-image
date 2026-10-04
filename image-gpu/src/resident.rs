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

//! GPU-RESIDENT textures and a recording batch over them.
//!
//! The single-tile lanes (`execute`) upload their inputs, dispatch once,
//! and read the output straight back: one submit and one readback per
//! dispatch. That is the simplest correct realization of the ABI, and
//! it is what a chain of N dispatches cannot afford — every
//! intermediate result crosses to the CPU and back.
//!
//! [`GpuBatch`] records ANY number of dispatches, texture-to-texture
//! copies and readbacks into ONE command encoder, submits once, and maps
//! ONE buffer holding every requested readback. A dispatch writes a
//! texture the next dispatch reads, so intermediates never leave the
//! device. [`Resident`] is a texture that outlives a batch, so a caller
//! can keep plates, accumulators and checkpoints on the GPU between
//! operations.
//!
//! The recorded dispatch is EXACTLY the single-tile lanes' dispatch:
//! the same cached pipeline, the same four bind groups, the same
//! workgroup grid sized by the output. Only where the inputs come from
//! differs (a texture written on the device instead of the same f16
//! bytes uploaded again), and rgba16float holds those bytes losslessly
//! either way, so a chain through a batch is byte-equal to the same
//! chain through `execute_tile_once`.
//!
//! # Ordering
//!
//! Uploads go through `queue.write_texture`, which wgpu stages AHEAD of
//! the next submitted command buffer. An upload therefore always lands
//! in a texture no earlier command of the same batch touches: uploads
//! only ever target textures freshly taken from the scratch pool, and a
//! texture returns to the pool only once every handle — including the
//! one each batch keeps for everything it recorded — is dropped.
//! Readbacks are recorded at the END of the batch, so a read observes
//! the texture's final state in that batch.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use image_kernels::{abi, KernelDef};

use crate::{GpuContext, GpuError};

/// The two texel formats the ABI binds: rgba16float images and r16float
/// masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TexFormat {
    /// rgba16float — kernel inputs and outputs (8 bytes per texel).
    Rgba16Float,
    /// r16float — the `@group(2)` mask (2 bytes per texel).
    R16Float,
}

impl TexFormat {
    fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            TexFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            TexFormat::R16Float => wgpu::TextureFormat::R16Float,
        }
    }

    /// Bytes per texel.
    pub fn bytes_per_texel(self) -> u32 {
        match self {
            TexFormat::Rgba16Float => 8,
            TexFormat::R16Float => 2,
        }
    }

    fn usage(self) -> wgpu::TextureUsages {
        let base = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        match self {
            TexFormat::Rgba16Float => base | wgpu::TextureUsages::STORAGE_BINDING,
            TexFormat::R16Float => base,
        }
    }
}

/// How many bytes of RELEASED textures the scratch pool keeps for
/// reuse. Past this a released texture is destroyed instead; the pool
/// is a reuse cache, not an allocator, so the bound only caps idle
/// memory.
const SCRATCH_IDLE_BYTES: u64 = 256 << 20;

/// Released textures, by size and format, ready to be handed out again
/// without a `create_texture`.
#[derive(Default)]
pub(crate) struct ScratchPool {
    free: HashMap<(u32, u32, TexFormat), Vec<wgpu::Texture>>,
    idle_bytes: u64,
}

impl ScratchPool {
    fn take(&mut self, w: u32, h: u32, format: TexFormat) -> Option<wgpu::Texture> {
        let tex = self.free.get_mut(&(w, h, format))?.pop()?;
        self.idle_bytes -= texture_bytes(w, h, format);
        Some(tex)
    }

    fn give(&mut self, w: u32, h: u32, format: TexFormat, tex: wgpu::Texture) {
        let bytes = texture_bytes(w, h, format);
        if self.idle_bytes + bytes > SCRATCH_IDLE_BYTES {
            return; // dropped: the texture is destroyed
        }
        self.idle_bytes += bytes;
        self.free.entry((w, h, format)).or_default().push(tex);
    }
}

fn texture_bytes(w: u32, h: u32, format: TexFormat) -> u64 {
    w as u64 * h as u64 * format.bytes_per_texel() as u64
}

struct Inner {
    tex: Option<wgpu::Texture>,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
    format: TexFormat,
    /// Where the texture goes when the last handle drops; `None` for
    /// textures that are simply destroyed (the shared constants).
    pool: Option<Arc<Mutex<ScratchPool>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let (Some(tex), Some(pool)) = (self.tex.take(), self.pool.as_ref()) {
            if let Ok(mut p) = pool.lock() {
                p.give(self.w, self.h, self.format, tex);
            }
        }
    }
}

/// A texture that lives on the device between batches. Cloning shares
/// it; the texture returns to the scratch pool when the last handle
/// drops. Never written after it was first produced, except through a
/// batch that owns it — so sharing a handle shares an immutable value.
#[derive(Clone)]
pub struct Resident(Arc<Inner>);

impl Resident {
    pub fn width(&self) -> u32 {
        self.0.w
    }

    pub fn height(&self) -> u32 {
        self.0.h
    }

    pub fn format(&self) -> TexFormat {
        self.0.format
    }

    /// Device bytes this texture holds.
    pub fn bytes(&self) -> u64 {
        texture_bytes(self.0.w, self.0.h, self.0.format)
    }

    pub(crate) fn texture(&self) -> &wgpu::Texture {
        self.0.tex.as_ref().expect("live resident texture")
    }

    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.0.view
    }
}

fn create_texture(ctx: &GpuContext, w: u32, h: u32, format: TexFormat) -> wgpu::Texture {
    crate::counters::bump(|c| c.textures_created += 1);
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resident"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.wgpu(),
        usage: format.usage(),
        view_formats: &[],
    })
}

fn wrap(
    tex: wgpu::Texture,
    w: u32,
    h: u32,
    format: TexFormat,
    pool: Option<Arc<Mutex<ScratchPool>>>,
) -> Resident {
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    Resident(Arc::new(Inner {
        tex: Some(tex),
        view,
        w,
        h,
        format,
        pool,
    }))
}

fn write_whole(ctx: &GpuContext, r: &Resident, bytes: &[u8]) {
    crate::counters::bump(|c| c.bytes_uploaded += bytes.len() as u64);
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: r.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(r.width() * r.format().bytes_per_texel()),
            rows_per_image: Some(r.height()),
        },
        wgpu::Extent3d {
            width: r.width(),
            height: r.height(),
            depth_or_array_layers: 1,
        },
    );
}

/// Which shared constant texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Constant {
    /// r16float 1.0 everywhere — the "no mask" binding.
    OneMask,
    /// rgba16float zero — transparent black.
    Zero,
}

/// How many constant textures a device keeps. Sizes vary with tile
/// edges and canvases; a handful covers a session, and past the bound
/// the set is simply rebuilt.
const MAX_CONSTANTS: usize = 32;

impl GpuContext {
    /// A scratch texture of the given size, reused from the pool when
    /// one is idle.
    pub fn scratch(&self, w: u32, h: u32, format: TexFormat) -> Resident {
        let reused = self
            .scratch
            .lock()
            .expect("scratch pool lock")
            .take(w, h, format);
        let tex = reused.unwrap_or_else(|| create_texture(self, w, h, format));
        wrap(tex, w, h, format, Some(Arc::clone(&self.scratch)))
    }

    /// Upload rgba16float (or r16float) texel bytes, tightly packed,
    /// into a fresh scratch texture.
    pub fn upload(&self, w: u32, h: u32, format: TexFormat, bytes: &[u8]) -> Resident {
        let r = self.scratch(w, h, format);
        write_whole(self, &r, bytes);
        r
    }

    /// A shared constant texture, built once per size and kept.
    pub(crate) fn constant(&self, which: Constant, w: u32, h: u32) -> Resident {
        let mut map = self.constants.lock().expect("constant cache lock");
        if let Some(r) = map.get(&(which, w, h)) {
            return r.clone();
        }
        if map.len() >= MAX_CONSTANTS {
            map.clear();
        }
        let (format, texel): (TexFormat, &[u8]) = match which {
            Constant::OneMask => (TexFormat::R16Float, &[0x00, 0x3C]),
            Constant::Zero => (TexFormat::Rgba16Float, &[0; 8]),
        };
        let bytes: Vec<u8> = texel
            .iter()
            .copied()
            .cycle()
            .take(texture_bytes(w, h, format) as usize)
            .collect();
        let r = wrap(create_texture(self, w, h, format), w, h, format, None);
        write_whole(self, &r, &bytes);
        map.insert((which, w, h), r.clone());
        r
    }

    /// The constant-1 r16float mask at `w`×`h` (the "no mask" binding),
    /// built once per size.
    pub fn one_mask(&self, w: u32, h: u32) -> Resident {
        self.constant(Constant::OneMask, w, h)
    }

    /// Transparent black rgba16float at `w`×`h`, built once per size.
    pub fn zeros(&self, w: u32, h: u32) -> Resident {
        self.constant(Constant::Zero, w, h)
    }
}

/// A readback recorded into a batch: which texture, which window, and
/// where its rows land in the batch's one mapped buffer.
struct ReadSpec {
    tex: Resident,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    offset: u64,
    row_bytes: u32,
    padded_row: u32,
}

/// One command encoder's worth of GPU work: dispatches, copies and
/// readbacks, submitted together.
pub struct GpuBatch<'c> {
    ctx: &'c GpuContext,
    encoder: wgpu::CommandEncoder,
    /// Every texture the batch recorded a command against, held until
    /// the batch has been submitted so none returns to the pool (and is
    /// re-uploaded into) while a recorded command still reads it.
    keep: Vec<Resident>,
    reads: Vec<ReadSpec>,
    read_bytes: u64,
    recorded: bool,
}

/// A handle to one readback of a batch: index into
/// [`GpuBatch::finish`]'s result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadTicket(pub usize);

impl<'c> GpuBatch<'c> {
    pub fn new(ctx: &'c GpuContext) -> Self {
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident batch"),
            });
        GpuBatch {
            ctx,
            encoder,
            keep: Vec::new(),
            reads: Vec::new(),
            read_bytes: 0,
            recorded: false,
        }
    }

    pub fn ctx(&self) -> &'c GpuContext {
        self.ctx
    }

    /// Upload into a fresh scratch texture (see [`GpuContext::upload`]).
    pub fn upload(&mut self, w: u32, h: u32, format: TexFormat, bytes: &[u8]) -> Resident {
        self.ctx.upload(w, h, format, bytes)
    }

    /// A fresh rgba16float scratch texture to dispatch into.
    pub fn target(&mut self, w: u32, h: u32) -> Resident {
        self.ctx.scratch(w, h, TexFormat::Rgba16Float)
    }

    /// Record one dispatch of `def`: `inputs` (arity per the kernel)
    /// bound at group0, `params` at group1, `mask` at group2 (`None` =
    /// the constant-1 mask), `out` at group3. The workgroup grid is
    /// sized by `out` — exactly the single-tile lanes' dispatch.
    pub fn dispatch(
        &mut self,
        def: &'static KernelDef,
        inputs: &[&Resident],
        params: &[u8],
        mask: Option<&Resident>,
        out: &Resident,
    ) -> Result<(), GpuError> {
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
        let (w, h) = (out.width(), out.height());
        let pipeline = self.ctx.pipeline(def);
        let mask = match mask {
            Some(m) => m.clone(),
            None => self.ctx.one_mask(w, h),
        };
        let device = &self.ctx.device;

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(def.id),
            size: def.params.size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&params_buf, 0, params);
        crate::counters::bump(|c| c.bytes_uploaded += params.len() as u64);

        let g0_entries: Vec<wgpu::BindGroupEntry> = inputs
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(r.view()),
            })
            .collect();
        let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g0"),
            layout: &pipeline.group0,
            entries: &g0_entries,
        });
        let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g1"),
            layout: &pipeline.group1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });
        let g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g2"),
            layout: &pipeline.group2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(mask.view()),
            }],
        });
        let g3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g3"),
            layout: &pipeline.group3,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(out.view()),
            }],
        });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
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
        crate::counters::bump(|c| {
            c.dispatches += 1;
            c.dispatched_texels += w as u64 * h as u64;
        });
        self.keep.extend(inputs.iter().map(|r| (*r).clone()));
        self.keep.push(mask);
        self.keep.push(out.clone());
        self.recorded = true;
        Ok(())
    }

    /// Record a texel copy of the `w`×`h` window at `src_xy` in `src`
    /// to `dst_xy` in `dst` (same format).
    pub fn copy(
        &mut self,
        src: &Resident,
        src_xy: (u32, u32),
        dst: &Resident,
        dst_xy: (u32, u32),
        w: u32,
        h: u32,
    ) {
        self.encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: src_xy.0,
                    y: src_xy.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: dst_xy.0,
                    y: dst_xy.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.keep.push(src.clone());
        self.keep.push(dst.clone());
        self.recorded = true;
    }

    /// Request the whole of `tex` back on the CPU when the batch
    /// finishes (tightly packed rows).
    pub fn read(&mut self, tex: &Resident) -> ReadTicket {
        self.read_window(tex, 0, 0, tex.width(), tex.height())
    }

    /// Request the `w`×`h` window at (`x`, `y`) of `tex`.
    pub fn read_window(&mut self, tex: &Resident, x: u32, y: u32, w: u32, h: u32) -> ReadTicket {
        let row_bytes = w * tex.format().bytes_per_texel();
        let padded_row = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let offset = self.read_bytes;
        self.read_bytes += padded_row as u64 * h as u64;
        self.reads.push(ReadSpec {
            tex: tex.clone(),
            x,
            y,
            w,
            h,
            offset,
            row_bytes,
            padded_row,
        });
        ReadTicket(self.reads.len() - 1)
    }

    /// Record the readback copies, submit once, and return the pending
    /// map (one buffer for every read).
    fn submit(mut self) -> PendingReads {
        let buffer = if self.reads.is_empty() {
            None
        } else {
            let buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("resident readback"),
                size: self.read_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            for r in &self.reads {
                self.encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: r.tex.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: r.x,
                            y: r.y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: r.offset,
                            bytes_per_row: Some(r.padded_row),
                            rows_per_image: Some(r.h),
                        },
                    },
                    wgpu::Extent3d {
                        width: r.w,
                        height: r.h,
                        depth_or_array_layers: 1,
                    },
                );
            }
            Some(buffer)
        };
        if self.recorded || buffer.is_some() {
            self.ctx.queue.submit([self.encoder.finish()]);
            crate::counters::bump(|c| c.submits += 1);
        }
        let layout = self
            .reads
            .iter()
            .map(|r| (r.offset, r.row_bytes, r.padded_row, r.h))
            .collect();
        PendingReads {
            buffer,
            layout,
            size: self.read_bytes,
            _keep: std::mem::take(&mut self.keep),
        }
    }

    /// Submit and read back synchronously (native: a blocking poll).
    pub fn finish(self) -> Result<Vec<Vec<u8>>, GpuError> {
        let ctx = self.ctx;
        let pending = self.submit();
        let Some(buffer) = pending.buffer.as_ref() else {
            return Ok(Vec::new());
        };
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| GpuError::Readback("map callback dropped".into()))?
            .map_err(|e| GpuError::Readback(format!("map_async: {e:?}")))?;
        Ok(pending.collect())
    }

    /// Submit and read back ASYNCHRONOUSLY — the wasm32/WebGPU lane,
    /// where the browser event loop delivers the map callback. Natively
    /// the blocking poll still pumps it, so the await is ready at once.
    pub async fn finish_async(self) -> Result<Vec<Vec<u8>>, GpuError> {
        let ctx = self.ctx;
        let pending = self.submit();
        let Some(buffer) = pending.buffer.as_ref() else {
            return Ok(Vec::new());
        };
        let slice = buffer.slice(..);
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
        Ok(pending.collect())
    }
}

/// A submitted batch whose single readback buffer awaits its map.
struct PendingReads {
    buffer: Option<wgpu::Buffer>,
    /// Per read: (offset, row bytes, padded row bytes, rows).
    layout: Vec<(u64, u32, u32, u32)>,
    size: u64,
    _keep: Vec<Resident>,
}

impl PendingReads {
    fn collect(&self) -> Vec<Vec<u8>> {
        let Some(buffer) = self.buffer.as_ref() else {
            return Vec::new();
        };
        crate::counters::bump(|c| {
            c.readbacks += 1;
            c.bytes_read_back += self.size;
        });
        let mut out = Vec::with_capacity(self.layout.len());
        {
            let data = buffer.slice(..).get_mapped_range();
            for &(offset, row_bytes, padded_row, h) in &self.layout {
                let mut bytes = Vec::with_capacity(row_bytes as usize * h as usize);
                for row in 0..h as u64 {
                    let start = (offset + row * padded_row as u64) as usize;
                    bytes.extend_from_slice(&data[start..start + row_bytes as usize]);
                }
                out.push(bytes);
            }
        }
        buffer.unmap();
        out
    }
}

/// The map-callback rendezvous (see `execute`'s twin).
#[derive(Default)]
struct MapShared {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

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
