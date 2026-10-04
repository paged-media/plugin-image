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

//! RESIDENT CHAINS: several kernels over several tiles in one submit.
//!
//! A pipeline pull used to dispatch every stage of every tile on its
//! own — upload, dispatch, submit, read back — so a chain of N point
//! stages over T tiles cost N·T submits and N·T readbacks, and every
//! intermediate result crossed to the CPU and back. A chain keeps the
//! intermediates in ping-pong textures on the device: each tile's
//! inputs are uploaded once, stage k reads stage k−1's output texture,
//! and only the LAST stage of each tile is read back. Every tile of the
//! call shares one encoder, one submit and one mapped buffer.
//!
//! Byte-equal to running the same stages one `execute_tile_once` at a
//! time: each dispatch is the identical recorded dispatch (see
//! `resident`), and an rgba16float texture carries exactly the f16
//! bytes the round trip through the CPU would have re-uploaded.

use image_kernels::KernelDef;

use crate::resident::{GpuBatch, TexFormat};
use crate::{GpuContext, GpuError};

/// One stage of a chain: a kernel and its param block.
#[derive(Clone, Copy)]
pub struct ChainStage<'a> {
    pub def: &'static KernelDef,
    pub params: &'a [u8],
}

/// One tile of a chain: the FIRST stage's inputs (rgba16float bytes,
/// arity per that kernel), an optional r16float mask shared by every
/// stage (`None` = the constant-1 mask), and the dispatch extent.
pub struct ChainTile<'a> {
    pub inputs: Vec<&'a [u8]>,
    pub mask: Option<&'a [u8]>,
    pub w: u32,
    pub h: u32,
}

fn validate(stages: &[ChainStage<'_>], tiles: &[ChainTile<'_>]) -> Result<(), GpuError> {
    let Some(first) = stages.first() else {
        return Err(GpuError::Kernel {
            kernel: "chain",
            detail: "a chain needs at least one stage".into(),
        });
    };
    if let Some(s) = stages[1..].iter().find(|s| s.def.inputs != 1) {
        return Err(GpuError::Kernel {
            kernel: s.def.id,
            detail: "only the first stage of a chain may take more than one input".into(),
        });
    }
    if let Some(t) = tiles
        .iter()
        .find(|t| t.inputs.len() != first.def.inputs as usize)
    {
        return Err(GpuError::Kernel {
            kernel: first.def.id,
            detail: format!(
                "expected {} inputs, got {}",
                first.def.inputs,
                t.inputs.len()
            ),
        });
    }
    Ok(())
}

/// Record the whole chain into `batch`; returns one ticket per tile.
fn record(
    batch: &mut GpuBatch<'_>,
    stages: &[ChainStage<'_>],
    tiles: &[ChainTile<'_>],
) -> Result<Vec<crate::ReadTicket>, GpuError> {
    let mut tickets = Vec::with_capacity(tiles.len());
    for t in tiles {
        let inputs: Vec<_> = t
            .inputs
            .iter()
            .map(|b| batch.upload(t.w, t.h, TexFormat::Rgba16Float, b))
            .collect();
        let mask = t
            .mask
            .map(|m| batch.upload(t.w, t.h, TexFormat::R16Float, m));
        let mut cur = batch.target(t.w, t.h);
        let refs: Vec<_> = inputs.iter().collect();
        batch.dispatch(stages[0].def, &refs, stages[0].params, mask.as_ref(), &cur)?;
        for s in &stages[1..] {
            let next = batch.target(t.w, t.h);
            batch.dispatch(s.def, &[&cur], s.params, mask.as_ref(), &next)?;
            cur = next;
        }
        tickets.push(batch.read(&cur));
    }
    Ok(tickets)
}

/// Run `stages` over every tile in ONE submit with ONE readback (native,
/// blocking). Returns each tile's final rgba16float bytes, in order.
pub fn execute_chain(
    ctx: &GpuContext,
    stages: &[ChainStage<'_>],
    tiles: &[ChainTile<'_>],
) -> Result<Vec<Vec<u8>>, GpuError> {
    validate(stages, tiles)?;
    if tiles.is_empty() {
        return Ok(Vec::new());
    }
    let mut batch = GpuBatch::new(ctx);
    record(&mut batch, stages, tiles)?;
    batch.finish()
}

/// [`execute_chain`]'s ASYNC twin (the wasm32/WebGPU readback lane).
pub async fn execute_chain_async(
    ctx: &GpuContext,
    stages: &[ChainStage<'_>],
    tiles: &[ChainTile<'_>],
) -> Result<Vec<Vec<u8>>, GpuError> {
    validate(stages, tiles)?;
    if tiles.is_empty() {
        return Ok(Vec::new());
    }
    let mut batch = GpuBatch::new(ctx);
    record(&mut batch, stages, tiles)?;
    batch.finish_async().await
}
