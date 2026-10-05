// Small CMYK PSDs with REAL merged data and an EMBEDDED profile, written
// by Photoshop itself, plus Photoshop's own RGB conversion of each
// document as PNGs. The replay (psd_cmyk_photoshop.rs) answers three
// questions from them:
//
//  * how a CMYK PSD stores its ink (the `patches` case fills known ink
//    percentages and records them);
//  * which conversion Photoshop's RGB view of a CMYK document is (the
//    merged document converted to sRGB under three intent/BPC settings,
//    compared with our transform of the same merged CMYK numbers);
//  * how far compositing AFTER the conversion (our layer stack is RGB)
//    lands from Photoshop compositing in CMYK and converting the result
//    (normal stacks with opacity and soft edges; an opaque hard-edged
//    stack; one tile per non-normal blend mode; a 16-bit document).
//
// The profile is named explicitly (Coated FOGRA39, the CMYK working space
// of "Europe General Purpose 3") so the recording does not depend on the
// user's Color Settings, which are read and never changed. "Maximize
// compatibility" is set for the run and restored afterwards. Every
// conversion runs on a DUPLICATE whose layers were merged first, so it
// converts exactly the CMYK numbers of the merged composite; dither is
// off so the conversion is deterministic.
(function () {
  var P = PagedProbe;
  var st = P.begin("cmyk-stacks");
  var PROFILE = "Coated FOGRA39 (ISO 12647-2:2004)";
  var SRGB = "sRGB IEC61966-2.1";
  var prevCompat = app.preferences.maximizeCompatibility;
  app.preferences.maximizeCompatibility = QueryStateType.ALWAYS;
  st.maximize_compatibility_was = String(prevCompat);
  st.findings = { cmyk_profile: PROFILE };

  // The conversion Photoshop's Color Settings would use (Edit > Convert
  // to Profile defaults to it). Read through the ActionManager; recorded
  // as found, never changed.
  st.findings.conversion_settings = (function () {
    try {
      var ref = new ActionReference();
      ref.putProperty(P.cid("Prpr"), P.cid("colorSettings"));
      ref.putEnumerated(P.cid("capp"), P.cid("Ordn"), P.cid("Trgt"));
      var cs = executeActionGet(ref).getObjectValue(P.cid("colorSettings"));
      var out = {};
      var keys = ["engine", "intent", "mapBlack", "dither", "blendRGBGamma"];
      var i, k, t;
      for (i = 0; i < keys.length; i++) {
        k = P.cid(keys[i]);
        if (!cs.hasKey(k)) continue;
        t = cs.getType(k);
        if (t == DescValueType.BOOLEANTYPE) out[keys[i]] = cs.getBoolean(k);
        else if (t == DescValueType.ENUMERATEDTYPE) out[keys[i]] = typeIDToStringID(cs.getEnumerationValue(k));
        else if (t == DescValueType.STRINGTYPE) out[keys[i]] = cs.getString(k);
        else if (t == DescValueType.DOUBLETYPE) out[keys[i]] = cs.getDouble(k);
        else out[keys[i]] = "type " + String(t);
      }
      var all = [];
      for (i = 0; i < cs.count; i++) all.push(typeIDToStringID(cs.getKey(i)));
      out.keys = all.join(",");
      return out;
    } catch (e) {
      return "unavailable: " + e;
    }
  })();

  function cmyk(c, m, y, k) {
    var s = new SolidColor();
    s.cmyk.cyan = c;
    s.cmyk.magenta = m;
    s.cmyk.yellow = y;
    s.cmyk.black = k;
    return s;
  }
  function rect(d, x0, y0, x1, y1, color, feather) {
    d.selection.select([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    if (feather) d.selection.feather(feather);
    d.selection.fill(color);
    d.selection.deselect();
  }
  function ellipse(d, x0, y0, x1, y1, color, feather) {
    var pts = [];
    var i, t;
    for (i = 0; i < 48; i++) {
      t = (i / 48) * 2 * Math.PI;
      pts.push([(x0 + x1) / 2 + ((x1 - x0) / 2) * Math.cos(t), (y0 + y1) / 2 + ((y1 - y0) / 2) * Math.sin(t)]);
    }
    d.selection.select(pts);
    if (feather) d.selection.feather(feather);
    d.selection.fill(color);
    d.selection.deselect();
  }
  function layer(d, name) {
    var l = d.artLayers.add();
    l.name = name;
    return l;
  }
  function background(d) {
    d.activeLayer = d.artLayers[d.artLayers.length - 1];
  }
  function revealSelectionMask() {
    var desc = new ActionDescriptor();
    desc.putClass(P.cid("Nw  "), P.cid("Chnl"));
    var ref = new ActionReference();
    ref.putEnumerated(P.cid("Chnl"), P.cid("Chnl"), P.cid("Msk "));
    desc.putReference(P.cid("At  "), ref);
    desc.putEnumerated(P.cid("Usng"), P.cid("UsrM"), P.cid("RvlS"));
    executeAction(P.cid("Mk  "), desc, DialogModes.NO);
  }

  // The three RGB views: Photoshop's Color Settings default (relative
  // colorimetric + black-point compensation), and the two a CMM without
  // BPC can offer.
  var VIEWS = [
    ["relcol-bpc", Intent.RELATIVECOLORIMETRIC, true],
    ["relcol", Intent.RELATIVECOLORIMETRIC, false],
    ["perceptual", Intent.PERCEPTUAL, false]
  ];

  function convertViews(d, id, c) {
    var i, dup;
    c.views = [];
    for (i = 0; i < VIEWS.length; i++) {
      dup = null;
      try {
        dup = d.duplicate(id + "-" + VIEWS[i][0], false);
        app.activeDocument = dup;
        if (dup.layers.length > 1 || dup.layerSets.length > 0) dup.mergeVisibleLayers();
        dup.convertProfile(SRGB, VIEWS[i][1], VIEWS[i][2], false);
        var png = new PNGSaveOptions();
        png.compression = 9;
        var rel = "cmyk-stacks/" + id + "." + VIEWS[i][0] + ".png";
        dup.saveAs(new File(PAGED_STAGE + "/" + rel), png, true, Extension.LOWERCASE);
        c.files.push(rel);
        c.views.push({ view: VIEWS[i][0], profile: P.profileOf(dup), bits: dup.bitsPerChannel == BitsPerChannelType.SIXTEEN ? 16 : 8 });
      } finally {
        if (dup !== null) dup.close(SaveOptions.DONOTSAVECHANGES);
      }
    }
  }

  function doc(id, w, h, fill, bits, build) {
    var c = { id: id, params: {}, files: [] };
    var d = null;
    try {
      d = app.documents.add(w, h, 72, id, NewDocumentMode.CMYK, fill, 1.0,
        bits === 16 ? BitsPerChannelType.SIXTEEN : BitsPerChannelType.EIGHT, PROFILE);
      build(d, c.params);
      var psd = new PhotoshopSaveOptions();
      psd.layers = true;
      psd.embedColorProfile = true;
      psd.alphaChannels = true;
      var rel = "cmyk-stacks/" + id + ".psd";
      d.saveAs(new File(PAGED_STAGE + "/" + rel), psd, true, Extension.LOWERCASE);
      c.files.push(rel);
      c.profile = P.profileOf(d);
      c.bits = bits;
      convertViews(d, id, c);
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
    st.cases.push(c);
  }

  // Ink patches: an 8x8 grid of 8 px squares on the background, each a
  // KNOWN ink (percent), corners and solids first, then a deterministic
  // spread. One saved alpha channel rides along, so the replay can check
  // that an extra channel is not read as transparency.
  function patchInks() {
    var inks = [
      [0, 0, 0, 0], [100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 100, 0],
      [0, 0, 0, 100], [100, 100, 100, 100], [50, 50, 50, 50], [0, 0, 0, 50],
      [100, 100, 0, 0], [0, 100, 100, 0], [100, 0, 100, 0], [60, 40, 40, 100],
      [25, 0, 0, 0], [0, 25, 0, 0], [0, 0, 25, 0], [0, 0, 0, 25]
    ];
    var s = 12345;
    function next() {
      s = (s * 1103515245 + 12345) % 2147483648;
      return Math.floor((s / 2147483648) * 101);
    }
    while (inks.length < 64) inks.push([next(), next(), next(), next() % 60]);
    return inks;
  }

  try {
    doc("patches", 64, 64, DocumentFill.WHITE, 8, function (d, p) {
      var inks = patchInks();
      var i;
      background(d);
      for (i = 0; i < 64; i++) {
        rect(d, (i % 8) * 8, Math.floor(i / 8) * 8, (i % 8) * 8 + 8, Math.floor(i / 8) * 8 + 8,
          cmyk(inks[i][0], inks[i][1], inks[i][2], inks[i][3]), 0);
      }
      d.selection.select([[0, 0], [32, 0], [32, 32], [0, 32]]);
      d.channels.add();
      d.selection.deselect();
      p.what = "8x8 known-ink patches (percent, row-major, 8 px each) + one saved alpha channel";
      p.inks = inks;
    });
    doc("transparent-soft", 64, 64, DocumentFill.TRANSPARENT, 8, function (d, p) {
      ellipse(d, 8, 8, 56, 56, cmyk(10, 80, 90, 0), 6);
      p.what = "one layer, feathered ellipse over transparency";
    });
    function normalStack(d, p) {
      background(d);
      rect(d, 0, 0, 64, 32, cmyk(70, 15, 0, 0), 0);
      var a = layer(d, "normal 60");
      rect(d, 16, 8, 56, 56, cmyk(0, 30, 95, 0), 0);
      a.opacity = 60;
      layer(d, "soft");
      ellipse(d, 4, 20, 44, 60, cmyk(75, 0, 90, 10), 4);
      layer(d, "masked");
      rect(d, 30, 30, 64, 64, cmyk(20, 90, 0, 0), 0);
      d.selection.select([[36, 36], [60, 36], [60, 60], [36, 60]]);
      d.selection.feather(4);
      revealSelectionMask();
      d.selection.deselect();
      var g = d.layerSets.add();
      g.name = "group 50";
      g.opacity = 50;
      g.blendMode = BlendMode.NORMAL;
      var g1 = g.artLayers.add();
      d.activeLayer = g1;
      rect(d, 2, 2, 30, 26, cmyk(0, 0, 0, 90), 0);
      var g2 = g.artLayers.add();
      d.activeLayer = g2;
      rect(d, 14, 10, 40, 30, cmyk(0, 100, 60, 0), 0);
      p.what = "normal blending only: background, a 60% layer, a feathered layer, a feather-masked layer, a 50% group of two";
    }
    doc("normal-stack", 64, 64, DocumentFill.WHITE, 8, normalStack);
    doc("normal-stack-16", 64, 64, DocumentFill.WHITE, 16, normalStack);
    // 256 px, so the disc's anti-aliased rim is the fraction of the
    // canvas it is in a real document (~1 %), not a 64 px thumbnail's.
    doc("opaque-stack", 256, 256, DocumentFill.WHITE, 8, function (d, p) {
      background(d);
      rect(d, 0, 0, 256, 128, cmyk(70, 15, 0, 0), 0);
      layer(d, "rect a");
      rect(d, 32, 32, 160, 160, cmyk(0, 30, 95, 0), 0);
      layer(d, "rect b");
      rect(d, 96, 80, 240, 224, cmyk(60, 50, 40, 80), 0);
      layer(d, "disc");
      ellipse(d, 16, 120, 136, 240, cmyk(75, 0, 90, 10), 0);
      p.what = "normal blending, every layer at 100%, hard-edged rectangles and one anti-aliased disc (256 px)";
    });
    doc("blend-modes", 160, 128, DocumentFill.WHITE, 8, function (d, p) {
      var modes = [
        ["normal-100", BlendMode.NORMAL, 100], ["normal-50", BlendMode.NORMAL, 50],
        ["multiply", BlendMode.MULTIPLY, 100], ["screen", BlendMode.SCREEN, 100],
        ["overlay", BlendMode.OVERLAY, 100], ["darken", BlendMode.DARKEN, 100],
        ["lighten", BlendMode.LIGHTEN, 100], ["color-burn", BlendMode.COLORBURN, 100],
        ["color-dodge", BlendMode.COLORDODGE, 100], ["soft-light", BlendMode.SOFTLIGHT, 100],
        ["hard-light", BlendMode.HARDLIGHT, 100], ["difference", BlendMode.DIFFERENCE, 100],
        ["exclusion", BlendMode.EXCLUSION, 100], ["linear-burn", BlendMode.LINEARBURN, 100],
        ["linear-dodge", BlendMode.LINEARDODGE, 100], ["hue", BlendMode.HUE, 100],
        ["saturation", BlendMode.SATURATION, 100], ["color", BlendMode.COLORBLEND, 100],
        ["luminosity", BlendMode.LUMINOSITY, 100], ["multiply-50", BlendMode.MULTIPLY, 50]
      ];
      var i, x, y, l;
      background(d);
      for (i = 0; i < modes.length; i++) {
        x = (i % 5) * 32;
        y = Math.floor(i / 5) * 32;
        rect(d, x, y, x + 16, y + 32, cmyk(45, 5, 10, 0), 0);
        rect(d, x + 16, y, x + 32, y + 32, cmyk(0, 70, 90, 5), 0);
        rect(d, x + 16, y + 20, x + 32, y + 32, cmyk(60, 50, 40, 80), 0);
      }
      p.tiles = [];
      for (i = 0; i < modes.length; i++) {
        x = (i % 5) * 32;
        y = Math.floor(i / 5) * 32;
        l = layer(d, modes[i][0]);
        rect(d, x + 6, y + 6, x + 26, y + 26, cmyk(80, 10, 60, 0), 2);
        l.blendMode = modes[i][1];
        l.opacity = modes[i][2];
        p.tiles.push({ id: modes[i][0], x: x, y: y, size: 32 });
      }
      p.what = "5x4 tiles of 32 px: a three-ink background under one feathered layer per blend mode";
    });
  } finally {
    app.preferences.maximizeCompatibility = prevCompat;
  }
  return P.finish([]);
})();
