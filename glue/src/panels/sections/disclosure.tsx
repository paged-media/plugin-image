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

// Disclosure helpers for the image panel: a collapsible section, and the
// collapsed "About" block that holds a section's longer explanation so the
// panel itself reads as controls plus one plain sentence.

import type { ReactNode } from "react";

import { note, sectionTitle } from "./styles";

/** A collapsible panel section. Open by default: the controls inside are
 *  the panel's working surface, and collapsing is the user's choice. */
export function Section({
  title,
  open = true,
  children,
  ...data
}: {
  title: ReactNode;
  open?: boolean;
  children: ReactNode;
  [data: `data-${string}`]: string | boolean | undefined;
}) {
  return (
    <details open={open} {...data}>
      <summary style={{ ...sectionTitle, cursor: "pointer" }}>{title}</summary>
      {children}
    </details>
  );
}

/** The longer explanation behind a section, collapsed. */
export function About({ children }: { children: ReactNode }) {
  return (
    <details style={note}>
      <summary style={{ cursor: "pointer" }}>About</summary>
      {children}
    </details>
  );
}
