// GROUP MASKS, written by Photoshop itself: a mask on a pass-through
// group, on isolated groups at 100 % and 60 %, switched off, combined with
// a member's own mask, and nested. Each is a layered PSD with REAL merged
// data plus a PNG of the same document; the replay is psd_group_masks.rs.
//
// "Maximize compatibility" is what makes Photoshop write the real merged
// composite; the preference is set for the run and restored after.
(function () {
  var P = PagedProbe;
  var st = P.begin("group-masks");
  var S = 64;
  var prevCompat = app.preferences.maximizeCompatibility;
  app.preferences.maximizeCompatibility = QueryStateType.ALWAYS;
  st.maximize_compatibility_was = String(prevCompat);

  function rgb(r, g, b) {
    var c = new SolidColor();
    c.rgb.red = r;
    c.rgb.green = g;
    c.rgb.blue = b;
    return c;
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

  function stack(id, fill, build) {
    var c = { id: id, params: {} };
    var d = null;
    try {
      d = app.documents.add(S, S, 72, id, NewDocumentMode.RGB, fill);
      build(d, c.params);
      var psd = new PhotoshopSaveOptions();
      psd.layers = true;
      psd.embedColorProfile = false;
      psd.alphaChannels = false;
      d.saveAs(new File(PAGED_STAGE + "/group-masks/" + id + ".psd"), psd, true, Extension.LOWERCASE);
      var png = new PNGSaveOptions();
      png.compression = 9;
      d.saveAs(new File(PAGED_STAGE + "/group-masks/" + id + ".png"), png, true, Extension.LOWERCASE);
      c.files = ["group-masks/" + id + ".psd", "group-masks/" + id + ".png"];
      c.profile = P.profileOf(d);
      c.bits = 8;
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

  function maskActive(x0, y0, x1, y1, feather) {
    var d = app.activeDocument;
    d.selection.select([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    if (feather) d.selection.feather(feather);
    revealSelectionMask();
    d.selection.deselect();
  }
  function disableMask() {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putEnumerated(P.cid("Lyr "), P.cid("Ordn"), P.cid("Trgt"));
    desc.putReference(P.cid("null"), ref);
    var to = new ActionDescriptor();
    to.putBoolean(P.cid("UsrM"), false);
    desc.putObject(P.cid("T   "), P.cid("Lyr "), to);
    executeAction(P.cid("setd"), desc, DialogModes.NO);
  }
  function groupWithTwo(d, name, opacity, passThrough) {
    var g = d.layerSets.add();
    g.name = name;
    if (!passThrough) g.blendMode = BlendMode.NORMAL;
    if (opacity !== 100) g.opacity = opacity;
    var a = g.artLayers.add();
    d.activeLayer = a;
    rect(d, 4, 4, 44, 44, rgb(60, 140, 220), 0);
    var b = g.artLayers.add();
    d.activeLayer = b;
    ellipse(d, 20, 20, 60, 60, rgb(200, 30, 30), 0);
    b.blendMode = BlendMode.MULTIPLY;
    d.activeLayer = g;
    return g;
  }

  try {
    stack("pass-through-masked", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 32, 64, rgb(250, 200, 30), 0);
      groupWithTwo(d, "group", 100, true);
      maskActive(12, 12, 52, 52, 4);
      p.what = "pass-through group (normal + multiply member) with a feathered mask";
    });
    stack("isolated-masked", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 32, 64, rgb(250, 200, 30), 0);
      groupWithTwo(d, "group", 100, false);
      maskActive(12, 12, 52, 52, 4);
      p.what = "isolated normal group with a feathered mask";
    });
    stack("isolated-60-masked", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 32, 64, rgb(250, 200, 30), 0);
      groupWithTwo(d, "group", 60, false);
      maskActive(8, 20, 56, 44, 0);
      p.what = "isolated group at 60% with a hard mask";
    });
    stack("group-mask-disabled", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 32, 64, rgb(250, 200, 30), 0);
      groupWithTwo(d, "group", 100, true);
      maskActive(12, 12, 52, 52, 0);
      disableMask();
      p.what = "pass-through group whose mask is switched off";
    });
    stack("group-and-member-masks", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 32, 64, rgb(250, 200, 30), 0);
      var g = groupWithTwo(d, "group", 100, true);
      d.activeLayer = g.artLayers[0];
      maskActive(0, 30, 64, 64, 3);
      d.activeLayer = g;
      maskActive(10, 0, 50, 64, 0);
      p.what = "member mask (bottom half) inside a group mask (middle band)";
    });
    stack("nested-masked", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 32, 64, rgb(250, 200, 30), 0);
      var outer = d.layerSets.add();
      outer.name = "outer";
      var inner = outer.layerSets.add();
      inner.name = "inner";
      var a = inner.artLayers.add();
      d.activeLayer = a;
      ellipse(d, 6, 6, 58, 58, rgb(60, 140, 220), 0);
      d.activeLayer = inner;
      maskActive(0, 0, 40, 64, 0);
      d.activeLayer = outer;
      maskActive(0, 16, 64, 48, 2);
      p.what = "a masked group inside a masked group";
    });
  } finally {
    app.preferences.maximizeCompatibility = prevCompat;
  }
  return P.finish([]);
})();
