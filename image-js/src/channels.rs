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

//! The CHANNELS readout — per-channel statistics, and the one operation
//! a channel list exists to enable: **load a channel as the selection**.
//!
//! WHAT A CHANNELS PANEL IS FOR. In Photoshop it does three things:
//! shows the per-channel content, lets you view a channel in isolation,
//! and lets you turn a channel into a selection (the "load channel as
//! selection" that makes luminance masking possible). Only the third is
//! a data operation; the first is a readout, and the second is a VIEW
//! mode.
//!
//! This module ships the two that are honest here:
//!
//! * **statistics** — min, max and mean per channel, reduced from the
//!   same straight-RGBA8 buffer the histogram uses, so the numbers agree
//!   with the histogram by construction rather than by coincidence;
//! * **channel → coverage** — a channel's bytes ARE a mask, so this is a
//!   copy, not an estimate. It is why the panel is worth having: it turns
//!   the alpha channel of a PSD, or the blue channel of a scan, into a
//!   real selection the existing masked pipeline already honours.
//!
//! What it does NOT ship, and why: **isolated channel VIEW**. Showing
//! one channel alone on the canvas is a display state, and the host
//! scene channel takes a composited image, not a view mode — faking it
//! by writing a greyscale composite into the document would be a
//! destructive edit wearing a view's clothes. That belongs to a viewing
//! lane the contract does not have; the panel says so rather than
//! quietly doing the destructive thing.

/// A colour channel of the working RGBA8 buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Red,
    Green,
    Blue,
    Alpha,
    /// Not a stored channel — the Rec.709 luma of R/G/B, which is what
    /// "load luminosity as selection" means. Included because it is the
    /// most-used channel-as-mask and computing it in the panel would
    /// duplicate the coefficients that already live here.
    Luma,
}

impl Channel {
    /// Parse the panel's wire name. Returns `None` rather than defaulting
    /// to red — a typo'd channel must not silently mask on the wrong one.
    pub fn from_name(name: &str) -> Option<Channel> {
        Some(match name {
            "red" => Channel::Red,
            "green" => Channel::Green,
            "blue" => Channel::Blue,
            "alpha" => Channel::Alpha,
            "luma" => Channel::Luma,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Channel::Red => "red",
            Channel::Green => "green",
            Channel::Blue => "blue",
            Channel::Alpha => "alpha",
            Channel::Luma => "luma",
        }
    }

    /// The channel's value at one straight-RGBA8 pixel.
    fn value_at(self, px: &[u8]) -> u8 {
        match self {
            Channel::Red => px[0],
            Channel::Green => px[1],
            Channel::Blue => px[2],
            Channel::Alpha => px[3],
            // Rec.709 luma, rounded — the same coefficients the gradient
            // map uses, so "luma" means one thing across this crate.
            Channel::Luma => {
                (0.2126 * f32::from(px[0]) + 0.7152 * f32::from(px[1]) + 0.0722 * f32::from(px[2]))
                    .round()
                    .clamp(0.0, 255.0) as u8
            }
        }
    }
}

/// One channel's reduction over the whole buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelStats {
    pub min: u8,
    pub max: u8,
    /// The arithmetic mean, kept fractional — rounding it to a `u8` would
    /// make a nearly-flat channel read as exactly flat.
    pub mean: f64,
}

/// Reduce one channel of a straight-RGBA8 buffer.
///
/// An EMPTY buffer yields `None`, not a zeroed row: "no pixels" and "all
/// pixels are zero" are different, and a panel showing `0 / 0 / 0.0` for
/// the former would be stating a measurement it never made.
pub fn stats_of(rgba: &[u8], channel: Channel) -> Option<ChannelStats> {
    if rgba.len() < 4 {
        return None;
    }
    let mut min = u8::MAX;
    let mut max = u8::MIN;
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    for px in rgba.chunks_exact(4) {
        let v = channel.value_at(px);
        min = min.min(v);
        max = max.max(v);
        sum += u64::from(v);
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some(ChannelStats {
        min,
        max,
        mean: sum as f64 / count as f64,
    })
}

/// A channel's bytes, one per pixel — the payload of "load channel as
/// selection". This IS the coverage representation (u8 per pixel), so no
/// conversion or thresholding happens: a 50%-grey channel becomes a
/// 50%-selected region, which is what a luminosity mask means.
pub fn channel_bytes(rgba: &[u8], channel: Channel) -> Vec<u8> {
    rgba.chunks_exact(4)
        .map(|px| channel.value_at(px))
        .collect()
}

/// The four stored channels plus luma, as JSON for the panel:
/// `[{name, min, max, mean}]`.
pub fn stats_json(rgba: &[u8]) -> String {
    const ALL: [Channel; 5] = [
        Channel::Red,
        Channel::Green,
        Channel::Blue,
        Channel::Alpha,
        Channel::Luma,
    ];
    let rows: Vec<String> = ALL
        .iter()
        .map(|&c| match stats_of(rgba, c) {
            Some(s) => format!(
                "{{\"name\":\"{}\",\"min\":{},\"max\":{},\"mean\":{:.3}}}",
                c.name(),
                s.min,
                s.max,
                s.mean
            ),
            // An unmeasurable channel reports nulls rather than zeros.
            None => format!(
                "{{\"name\":\"{}\",\"min\":null,\"max\":null,\"mean\":null}}",
                c.name()
            ),
        })
        .collect();
    format!("[{}]", rows.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two pixels whose channels all differ, so a transposed read shows.
    const TWO: [u8; 8] = [10, 20, 30, 40, 200, 100, 50, 255];

    #[test]
    fn image_channels_readout_reduces_each_channel_separately() {
        assert_eq!(
            stats_of(&TWO, Channel::Red),
            Some(ChannelStats {
                min: 10,
                max: 200,
                mean: 105.0
            })
        );
        assert_eq!(
            stats_of(&TWO, Channel::Green),
            Some(ChannelStats {
                min: 20,
                max: 100,
                mean: 60.0
            })
        );
        assert_eq!(
            stats_of(&TWO, Channel::Blue),
            Some(ChannelStats {
                min: 30,
                max: 50,
                mean: 40.0
            })
        );
        assert_eq!(
            stats_of(&TWO, Channel::Alpha),
            Some(ChannelStats {
                min: 40,
                max: 255,
                mean: 147.5
            })
        );
    }

    /// "No pixels" is not "all pixels are zero".
    #[test]
    fn image_channels_readout_empty_is_unmeasured_not_zero() {
        assert_eq!(stats_of(&[], Channel::Red), None);
        assert!(stats_json(&[]).contains("\"min\":null"));
    }

    /// The channel bytes ARE the coverage — one byte per pixel, in order,
    /// with no thresholding. A mid-grey channel must stay mid-selected.
    #[test]
    fn image_channels_to_selection_copies_bytes_without_thresholding() {
        assert_eq!(channel_bytes(&TWO, Channel::Red), vec![10, 200]);
        assert_eq!(channel_bytes(&TWO, Channel::Alpha), vec![40, 255]);
        let grey = [128, 128, 128, 255];
        assert_eq!(
            channel_bytes(&grey, Channel::Red),
            vec![128],
            "a 50% channel is a 50% selection, not a rounded 0 or 255"
        );
    }

    /// Luma is Rec.709 and agrees with the gradient map's definition.
    #[test]
    fn image_channels_readout_luma_is_rec709() {
        // Pure green is the heaviest coefficient — a naive average would
        // give 85 for all three primaries and hide the difference.
        assert_eq!(channel_bytes(&[0, 255, 0, 255], Channel::Luma), vec![182]);
        assert_eq!(channel_bytes(&[255, 0, 0, 255], Channel::Luma), vec![54]);
        assert_eq!(channel_bytes(&[0, 0, 255, 255], Channel::Luma), vec![18]);
        assert_eq!(
            channel_bytes(&[255, 255, 255, 255], Channel::Luma),
            vec![255]
        );
    }

    /// A wrong channel name must NOT fall back to a working channel —
    /// masking on red when the caller asked for "rd" is a silent wrong
    /// answer, which is worse than an error.
    #[test]
    fn image_channels_to_selection_refuses_an_unknown_channel_name() {
        assert_eq!(Channel::from_name("red"), Some(Channel::Red));
        assert_eq!(Channel::from_name("luma"), Some(Channel::Luma));
        assert_eq!(Channel::from_name("rd"), None);
        assert_eq!(Channel::from_name(""), None);
        assert_eq!(
            Channel::from_name("Red"),
            None,
            "the wire name is lowercase"
        );
    }

    /// The JSON the panel parses, in a fixed order (the panel renders it
    /// as given — R, G, B, A, then the derived luma).
    #[test]
    fn image_channels_readout_json_lists_five_rows_in_order() {
        let json = stats_json(&TWO);
        let order: Vec<&str> = ["red", "green", "blue", "alpha", "luma"]
            .iter()
            .filter(|n| json.contains(&format!("\"name\":\"{n}\"")))
            .copied()
            .collect();
        assert_eq!(order, ["red", "green", "blue", "alpha", "luma"]);
        assert!(json.contains("\"mean\":105.000"), "{json}");
    }
}
