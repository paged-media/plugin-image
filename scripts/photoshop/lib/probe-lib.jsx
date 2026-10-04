// The shared half of every Photoshop oracle probe. `run-probe.sh`
// PREPENDS this file (after a one-line `var PAGED_STAGE = "...";`) to the
// probe it runs and hands Photoshop the concatenated text, so a probe
// needs no `#include`.
//
// LANGUAGE. ExtendScript -- ECMAScript 3. No `let`/`const`, no arrow
// functions, no `JSON`, no Array.prototype.map/forEach/indexOf, no
// trailing commas. ASCII only (the text crosses an Apple event).
//
// FILES. Photoshop reads the stimuli from, and writes every output to,
// the staging directory PAGED_STAGE (a mktemp dir under $TMPDIR, which
// Photoshop can read and write without a sandbox prompt). A probe
// returns JSON naming the files it wrote; the runner judges that reply
// and the files, never the exit code.
//
// COLOUR. The stimuli are UNTAGGED 16-bit PNGs. Opened with dialogs
// suppressed, an untagged file is interpreted in the working RGB space
// and no number is converted -- `begin` records the Color Settings and
// every probe carries an `identity` case (open, save) that the replay
// checks is the stimulus itself, which is what proves it.

var PagedProbe = (function () {
  function quote(s) {
    var out = '"';
    var i, c, code;
    s = String(s);
    for (i = 0; i < s.length; i++) {
      c = s.charAt(i);
      code = s.charCodeAt(i);
      if (c === '"') out += '\\"';
      else if (c === "\\") out += "\\\\";
      else if (c === "\n") out += "\\n";
      else if (c === "\r") out += "\\r";
      else if (c === "\t") out += "\\t";
      else if (code < 32 || code > 126) out += "\\u" + ("0000" + code.toString(16)).slice(-4);
      else out += c;
    }
    return out + '"';
  }

  function json(v) {
    var parts, i, k;
    if (v === null || v === undefined) return "null";
    if (typeof v === "number") return isFinite(v) ? String(v) : "null";
    if (typeof v === "boolean") return v ? "true" : "false";
    if (typeof v === "string") return quote(v);
    if (v instanceof Array) {
      parts = [];
      for (i = 0; i < v.length; i++) parts.push(json(v[i]));
      return "[" + parts.join(",") + "]";
    }
    parts = [];
    for (k in v) {
      if (v.hasOwnProperty(k)) parts.push(quote(k) + ":" + json(v[k]));
    }
    return "{" + parts.join(",") + "}";
  }

  function safe(f) {
    try {
      return String(f());
    } catch (e) {
      return "unavailable: " + e;
    }
  }

  var state = null;

  /** Start a probe: dialogs off, pixels as the unit, provenance read. */
  function begin(name) {
    app.displayDialogs = DialogModes.NO;
    app.preferences.rulerUnits = Units.PIXELS;
    state = {
      probe: name,
      app: {
        name: app.name,
        version: app.version,
        build: safe(function () {
          return app.build;
        }),
        locale: app.locale
      },
      color_settings: safe(function () {
        return app.colorSettings;
      }),
      documents_before: app.documents.length,
      stage: PAGED_STAGE,
      cases: []
    };
    return state;
  }

  function stagePath(rel) {
    return PAGED_STAGE + "/" + rel;
  }

  /** Open a staged file. Records the profile the document ended up with. */
  function open(rel) {
    var f = new File(stagePath(rel));
    if (!f.exists) throw new Error("missing stimulus " + rel);
    var d = app.open(f);
    if (d.bitsPerChannel !== BitsPerChannelType.SIXTEEN) {
      d.bitsPerChannel = BitsPerChannelType.SIXTEEN;
    }
    return d;
  }

  function profileOf(d) {
    return safe(function () {
      return d.colorProfileType == ColorProfile.NONE ? "none" : d.colorProfileName;
    });
  }

  /** Save `d` as an untagged-or-working-space 16-bit PNG copy. */
  function savePng(d, rel) {
    var o = new PNGSaveOptions();
    o.compression = 9;
    o.interlaced = false;
    d.saveAs(new File(stagePath(rel)), o, true, Extension.LOWERCASE);
    return rel;
  }

  /**
   * Run one case: `body(doc)` mutates a FRESH duplicate of `source`;
   * the result is flattened and saved as `<probe>/<id>.png`. A throw is
   * recorded on the case, never raised.
   */
  function run(id, source, params, body) {
    var c = { id: id, params: params };
    var d = null;
    try {
      d = source.duplicate(id, false);
      app.activeDocument = d;
      body(d);
      if (d.layers.length > 1) d.flatten();
      c.output = savePng(d, state.probe + "/" + id + ".png");
      c.profile = profileOf(d);
      c.bits = d.bitsPerChannel == BitsPerChannelType.SIXTEEN ? 16 : 8;
    } catch (e) {
      c.error = String(e) + (e.line ? " (line " + e.line + ")" : "");
    }
    if (d !== null) {
      try {
        d.close(SaveOptions.DONOTSAVECHANGES);
      } catch (e2) {
        c.close_error = String(e2);
      }
    }
    state.cases.push(c);
    return c;
  }

  /** Close the probe's own documents and hand back the reply. */
  function finish(owned) {
    var i;
    for (i = 0; i < owned.length; i++) {
      try {
        owned[i].close(SaveOptions.DONOTSAVECHANGES);
      } catch (e) {
        state.close_error = String(e);
      }
    }
    state.documents_after = app.documents.length;
    return json(state);
  }

  // -- ActionManager helpers --------------------------------------------

  function cid(s) {
    return s.length === 4 ? charIDToTypeID(s) : stringIDToTypeID(s);
  }

  /** Execute `event` with a descriptor built from [key, type, value]. */
  function exec(event, entries) {
    var d = new ActionDescriptor();
    var i, e;
    for (i = 0; i < entries.length; i++) {
      e = entries[i];
      if (e[1] === "int") d.putInteger(cid(e[0]), e[2]);
      else if (e[1] === "dbl") d.putDouble(cid(e[0]), e[2]);
      else if (e[1] === "bool") d.putBoolean(cid(e[0]), e[2]);
      else if (e[1] === "enum") d.putEnumerated(cid(e[0]), cid(e[2][0]), cid(e[2][1]));
      else if (e[1] === "unit") d.putUnitDouble(cid(e[0]), cid(e[2][0]), e[2][1]);
      else if (e[1] === "obj") d.putObject(cid(e[0]), cid(e[2][0]), e[2][1]);
      else if (e[1] === "list") d.putList(cid(e[0]), e[2]);
      else throw new Error("unknown AM type " + e[1]);
    }
    executeAction(cid(event), d, DialogModes.NO);
  }

  return {
    json: json,
    begin: begin,
    open: open,
    savePng: savePng,
    run: run,
    finish: finish,
    cid: cid,
    exec: exec,
    profileOf: profileOf
  };
})();
