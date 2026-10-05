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

// The colour row's treatment: the wire codes image-js `display_code`
// emits, mapped to names and labels. A CMYK source is CONVERTED, and the
// panel must say so rather than call it managed RGB.

import { describe, expect, it } from "vitest";

import { displayTreatmentLabel, displayTreatmentOf, isCmykConversion } from "../src/engine";

describe("display treatment", () => {
  it("maps every wire code, and an unknown one to the honest default", () => {
    expect(displayTreatmentOf(0)).toBe("managed");
    expect(displayTreatmentOf(1)).toBe("assumed-srgb");
    expect(displayTreatmentOf(2)).toBe("profile-rejected");
    expect(displayTreatmentOf(3)).toBe("cmyk-converted");
    expect(displayTreatmentOf(4)).toBe("cmyk-uncalibrated");
    expect(displayTreatmentOf(undefined)).toBe("assumed-srgb");
    expect(displayTreatmentOf(99)).toBe("assumed-srgb");
  });

  it("states a CMYK conversion as a conversion, and says when no profile drove it", () => {
    expect(isCmykConversion("cmyk-converted")).toBe(true);
    expect(isCmykConversion("cmyk-uncalibrated")).toBe(true);
    expect(isCmykConversion("managed")).toBe(false);
    expect(displayTreatmentLabel("cmyk-converted")).toMatch(/^CMYK, converted to sRGB$/);
    expect(displayTreatmentLabel("cmyk-uncalibrated")).toMatch(/without a profile/);
    expect(displayTreatmentLabel("managed")).toBe("ICC managed");
  });
});
