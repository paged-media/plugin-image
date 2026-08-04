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

//! A minimal, dependency-free JSON reader for the ANALYST-published
//! `corpus-profile.json` (spec §13.4 / §14.3.1).
//!
//! Why hand-rolled rather than `serde_json`: the profile is one small,
//! machine-generated document read by exactly one test lane, and
//! `image-conformance` is the crate whose dependency set the wasm
//! cargo-tree guard exists to keep honest. A ~200-line reader with its
//! own tests is cheaper to justify than a new dependency edge, and it
//! keeps the gate runnable on a bare checkout.
//!
//! Scope: the JSON the profile actually is — objects, arrays, strings
//! (with `\u` escapes; the provenance header carries `§`), numbers,
//! booleans and null. Nothing here is a general-purpose parser and it
//! does not try to be: it fails loudly rather than guessing.

/// One JSON value.
///
/// Objects keep their members as an ORDERED `Vec` rather than a map, for
/// the same reason [`image_psd::descriptor::Descriptor`] does: order is
/// information, and nothing should be silently dropped.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parse a whole document. Trailing non-whitespace is an error.
    pub fn parse(src: &str) -> Result<Json, String> {
        let mut p = Parser {
            b: src.as_bytes(),
            i: 0,
        };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("trailing bytes at offset {}", p.i));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// The value as a non-negative integer. Fractional or negative
    /// numbers yield `None` — the profile's counts are all counts.
    pub fn as_u64(&self) -> Option<u64> {
        let n = self.as_f64()?;
        (n >= 0.0 && n.fract() == 0.0).then_some(n as u64)
    }

    pub fn as_i64(&self) -> Option<i64> {
        let n = self.as_f64()?;
        (n.fract() == 0.0).then_some(n as i64)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Deepest nesting accepted — a guard, not a format limit (the profile
/// nests three deep).
const MAX_DEPTH: u32 = 32;

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            return Ok(());
        }
        Err(format!(
            "expected {:?} at offset {}, found {:?}",
            c as char,
            self.i,
            self.b.get(self.i).map(|c| *c as char)
        ))
    }

    fn value(&mut self, depth: u32) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(format!("nesting deeper than {MAX_DEPTH}"));
        }
        match self.b.get(self.i) {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(_) => self.number(),
            None => Err("unexpected end of document".into()),
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            return Ok(v);
        }
        Err(format!("expected `{word}` at offset {}", self.i))
    }

    fn object(&mut self, depth: u32) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Object(out));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value(depth + 1)?;
            out.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(out));
                }
                other => {
                    return Err(format!(
                        "expected `,` or `}}` at offset {}, found {:?}",
                        self.i,
                        other.map(|c| *c as char)
                    ))
                }
            }
        }
    }

    fn array(&mut self, depth: u32) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Array(out));
        }
        loop {
            self.ws();
            out.push(self.value(depth + 1)?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(out));
                }
                other => {
                    return Err(format!(
                        "expected `,` or `]` at offset {}, found {:?}",
                        self.i,
                        other.map(|c| *c as char)
                    ))
                }
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let c = *self
                .b
                .get(self.i)
                .ok_or_else(|| "unterminated string".to_string())?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self
                        .b
                        .get(self.i)
                        .ok_or_else(|| "unterminated escape".to_string())?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        other => {
                            return Err(format!(
                                "unknown escape `\\{}` at offset {}",
                                other as char,
                                self.i - 1
                            ))
                        }
                    }
                }
                // Everything else is copied through as UTF-8. The input
                // is a `&str`, so the bytes are valid by construction.
                _ => {
                    let start = self.i - 1;
                    while self.b.get(self.i).is_some_and(|c| c & 0xc0 == 0x80) {
                        self.i += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?,
                    );
                }
            }
        }
    }

    /// `\uXXXX`, including the surrogate-pair form. A lone surrogate
    /// becomes U+FFFD rather than an error — the profile has none, and a
    /// replacement character is a visible artifact where a panic is not.
    fn unicode_escape(&mut self) -> Result<char, String> {
        let hi = self.hex4()?;
        if !(0xd800..0xdc00).contains(&hi) {
            return Ok(char::from_u32(hi).unwrap_or('\u{fffd}'));
        }
        if self.b[self.i..].starts_with(b"\\u") {
            self.i += 2;
            let lo = self.hex4()?;
            if (0xdc00..0xe000).contains(&lo) {
                let c = 0x1_0000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                return Ok(char::from_u32(c).unwrap_or('\u{fffd}'));
            }
        }
        Ok('\u{fffd}')
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let s = self
            .b
            .get(self.i..self.i + 4)
            .ok_or_else(|| "truncated \\u escape".to_string())?;
        let s = std::str::from_utf8(s).map_err(|e| e.to_string())?;
        let v = u32::from_str_radix(s, 16).map_err(|e| e.to_string())?;
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self
            .b
            .get(self.i)
            .is_some_and(|c| matches!(c, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        s.parse::<f64>()
            .map(Json::Number)
            .map_err(|e| format!("bad number `{s}` at offset {start}: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_abr_corpus_json_reads_the_shapes_the_profile_uses() {
        let v =
            Json::parse(r#"{"a": 1, "b": [true, false, null, -2.5e2], "c": {"d": "x"}, "e": "y"}"#)
                .unwrap();
        assert_eq!(v.get("a").and_then(Json::as_u64), Some(1));
        let b = v.get("b").and_then(Json::as_array).unwrap();
        assert_eq!(b[0].as_bool(), Some(true));
        assert_eq!(b[1].as_bool(), Some(false));
        assert_eq!(b[2], Json::Null);
        assert_eq!(b[3].as_f64(), Some(-250.0));
        assert_eq!(
            v.get("c").and_then(|c| c.get("d")).and_then(Json::as_str),
            Some("x")
        );
        // A non-integral or negative number is not a count.
        assert_eq!(b[3].as_u64(), None);
    }

    #[test]
    fn image_abr_corpus_json_decodes_escapes_including_the_section_sign() {
        let v = Json::parse(r#"{"k": "§13.4 \"q\" \\ \n\tA"}"#).unwrap();
        assert_eq!(
            v.get("k").and_then(Json::as_str),
            Some("§13.4 \"q\" \\ \n\tA")
        );
        // A surrogate pair, and a lone surrogate that must not panic.
        assert_eq!(Json::parse(r#""😀""#).unwrap().as_str(), Some("😀"));
        assert_eq!(
            Json::parse(r#""\ud83d""#).unwrap().as_str(),
            Some("\u{fffd}")
        );
    }

    #[test]
    fn image_abr_corpus_json_rejects_rather_than_guesses() {
        for bad in [
            "{",
            "{\"a\"}",
            "[1,]",
            "{\"a\": }",
            "tru",
            "{} extra",
            "{\"a\": 1} {}",
        ] {
            assert!(Json::parse(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn image_abr_corpus_json_keeps_member_order_and_duplicate_keys() {
        let v = Json::parse(r#"{"b": 1, "a": 2, "b": 3}"#).unwrap();
        let m = v.as_object().unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].0, "b");
        assert_eq!(m[1].0, "a");
        // `get` returns the first, and nothing was dropped.
        assert_eq!(v.get("b").and_then(Json::as_u64), Some(1));
    }
}
