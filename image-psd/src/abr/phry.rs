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

//! The `phry` section — the brush-panel folder hierarchy.
//!
//! A 2,000-brush library is unusable as a flat list; the folder tree
//! **is** the product. The section decodes to a flattened open/close
//! token stream — not a nested structure — whose class ids name the node
//! kind, the same dispatch-by-class-id idiom the tips and tool options
//! use (behaviour spec §8.2 `[OBS]`).

use crate::descriptor::Descriptor;

use super::AbrWarning;

/// One node of the flattened hierarchy token stream, in document order.
#[derive(Debug, Clone, PartialEq)]
pub enum HierarchyNode {
    /// `Grup` — opens a folder.
    GroupOpen {
        /// `Nm  ` — an ordinary human label (`Buildings`, `Small Towns`).
        name: String,
        /// `zuid` — the folder's UUID.
        uid: Option<String>,
    },
    /// `preset` — one brush preset, a LEAF that carries **no payload at
    /// all**, so its link to the brush list is POSITIONAL (spec §8.2
    /// trap): the *n*-th `preset` node is the *n*-th element of the
    /// `desc` section's `Brsh` list.
    ///
    /// That is inference from a perfect count match across four files
    /// (720/720, 238/238, 2,053/2,053, 159/159), not a stated rule —
    /// there is simply nothing else it could be. When the counts do not
    /// match, `brush_index` stays `None` and the reader falls back to a
    /// flat list rather than mis-assigning brushes to folders.
    Preset { brush_index: Option<usize> },
    /// `groupEnd` — closes the most recently opened folder.
    GroupEnd,
    /// An unrecognised node class. Retained rather than dropped.
    Unknown { class_id: String },
}

/// Decode the `phry` descriptor into the node stream.
///
/// `phry` may legitimately be present with an EMPTY hierarchy list (one
/// fixture is exactly that while still having a brush), and is absent
/// altogether from 4 of 9 fixtures including files that do have brushes.
/// Present-and-empty means "no grouping"; absent means "flat". Neither
/// is an error.
pub(crate) fn decode(root: &Descriptor, warnings: &mut Vec<AbrWarning>) -> Vec<HierarchyNode> {
    let Some(list) = root.list(b"hierarchy") else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(list.len());
    for value in list {
        let Some(node) = value.as_descriptor() else {
            warnings.push(AbrWarning::UnexpectedClassId {
                context: "hierarchy element".into(),
                class_id: String::from_utf8_lossy(&value.ostype()).into_owned(),
            });
            continue;
        };
        let class = node.class_id.as_bytes();
        out.push(if class == b"Grup" {
            HierarchyNode::GroupOpen {
                name: node.text_display(b"Nm  ").unwrap_or_default().to_string(),
                uid: node.text(b"zuid").map(str::to_string),
            }
        } else if class == b"preset" {
            HierarchyNode::Preset { brush_index: None }
        } else if class == b"groupEnd" {
            HierarchyNode::GroupEnd
        } else {
            HierarchyNode::Unknown {
                class_id: node.class_id.text_lossy(),
            }
        });
    }
    out
}

/// Bind `preset` leaves to brush indices positionally, and check the
/// stream's integrity.
///
/// The open/close balance is a free integrity check: in all 5 corpus
/// files the counts match exactly and the running depth never goes
/// negative. An unbalanced hierarchy means something upstream was
/// mis-parsed — better asserted here than discovered later as a mangled
/// folder tree.
pub(crate) fn bind_and_check(
    nodes: &mut [HierarchyNode],
    brush_count: usize,
    warnings: &mut Vec<AbrWarning>,
) {
    let mut depth: i64 = 0;
    let mut min_depth: i64 = 0;
    let mut opens = 0usize;
    let mut closes = 0usize;
    let mut presets = 0usize;
    for n in nodes.iter() {
        match n {
            HierarchyNode::GroupOpen { .. } => {
                depth += 1;
                opens += 1;
            }
            HierarchyNode::GroupEnd => {
                depth -= 1;
                closes += 1;
                min_depth = min_depth.min(depth);
            }
            HierarchyNode::Preset { .. } => presets += 1,
            HierarchyNode::Unknown { .. } => {}
        }
    }
    if depth != 0 || min_depth < 0 {
        warnings.push(AbrWarning::HierarchyUnbalanced { opens, closes });
    }
    if nodes.is_empty() {
        // Present-and-empty: "no grouping", not an error, and it must
        // not hide the brushes.
        return;
    }
    if presets != brush_count {
        warnings.push(AbrWarning::HierarchyCountMismatch {
            presets,
            brushes: brush_count,
        });
        return;
    }
    let mut next = 0usize;
    for n in nodes.iter_mut() {
        if let HierarchyNode::Preset { brush_index } = n {
            *brush_index = Some(next);
            next += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream() -> Vec<HierarchyNode> {
        vec![
            HierarchyNode::GroupOpen {
                name: "Buildings".into(),
                uid: None,
            },
            HierarchyNode::Preset { brush_index: None },
            HierarchyNode::Preset { brush_index: None },
            HierarchyNode::GroupEnd,
            HierarchyNode::Preset { brush_index: None },
        ]
    }

    #[test]
    fn image_abr_phry_binds_presets_positionally() {
        let mut nodes = stream();
        let mut w = Vec::new();
        bind_and_check(&mut nodes, 3, &mut w);
        assert!(w.is_empty(), "{w:?}");
        let bound: Vec<_> = nodes
            .iter()
            .filter_map(|n| match n {
                HierarchyNode::Preset { brush_index } => Some(*brush_index),
                _ => None,
            })
            .collect();
        assert_eq!(bound, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn image_abr_phry_count_mismatch_falls_back_to_flat() {
        let mut nodes = stream();
        let mut w = Vec::new();
        bind_and_check(&mut nodes, 4, &mut w);
        assert!(matches!(
            w.as_slice(),
            [AbrWarning::HierarchyCountMismatch {
                presets: 3,
                brushes: 4
            }]
        ));
        // Nothing bound: mis-assigning brushes to folders is worse than
        // showing a flat list.
        assert!(nodes.iter().all(|n| !matches!(
            n,
            HierarchyNode::Preset {
                brush_index: Some(_)
            }
        )));
    }

    #[test]
    fn image_abr_phry_unbalanced_is_reported() {
        let mut nodes = vec![
            HierarchyNode::GroupEnd,
            HierarchyNode::Preset { brush_index: None },
        ];
        let mut w = Vec::new();
        bind_and_check(&mut nodes, 1, &mut w);
        assert!(w
            .iter()
            .any(|x| matches!(x, AbrWarning::HierarchyUnbalanced { .. })));
    }

    #[test]
    fn image_abr_phry_present_but_empty_is_not_an_error() {
        let mut nodes: Vec<HierarchyNode> = Vec::new();
        let mut w = Vec::new();
        bind_and_check(&mut nodes, 1, &mut w);
        assert!(w.is_empty(), "an empty hierarchy must not hide the brushes");
    }
}
