// Small layered RGB PSDs with REAL merged data, written by Photoshop
// itself, plus a PNG of the same document (straight alpha) as the
// colour reference. The replay (psd_composite_photoshop.rs) parses each
// PSD, checks resource 0x0421, flattens our layer stack and compares.
//
// "Maximize compatibility" is what makes Photoshop write the real merged
// composite; the preference is set for the run and restored after.
(function () {
  var P = PagedProbe;
  var st = P.begin("layer-stacks");
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
      d.saveAs(new File(PAGED_STAGE + "/layer-stacks/" + id + ".psd"), psd, true, Extension.LOWERCASE);
      var png = new PNGSaveOptions();
      png.compression = 9;
      d.saveAs(new File(PAGED_STAGE + "/layer-stacks/" + id + ".png"), png, true, Extension.LOWERCASE);
      c.files = ["layer-stacks/" + id + ".psd", "layer-stacks/" + id + ".png"];
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

  try {
    stack("transparent-soft", DocumentFill.TRANSPARENT, function (d, p) {
      ellipse(d, 8, 8, 56, 56, rgb(200, 60, 40), 6);
      p.what = "one layer, feathered ellipse over transparency";
    });
    stack("opacity-blend-stack", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 64, 32, rgb(30, 120, 200), 0);
      var a = layer(d, "multiply 70");
      rect(d, 16, 8, 56, 56, rgb(240, 180, 40), 0);
      a.blendMode = BlendMode.MULTIPLY;
      a.opacity = 70;
      var b = layer(d, "screen 100");
      ellipse(d, 4, 20, 44, 60, rgb(40, 200, 90), 3);
      b.blendMode = BlendMode.SCREEN;
      p.what = "background + multiply 70% + screen 100% (feathered)";
    });
    stack("clip-base-100", DocumentFill.WHITE, function (d, p) {
      var base = layer(d, "base");
      ellipse(d, 8, 8, 56, 56, rgb(40, 60, 80), 0);
      var c = layer(d, "clipped multiply");
      rect(d, 0, 0, 64, 64, rgb(128, 128, 128), 0);
      c.blendMode = BlendMode.MULTIPLY;
      c.grouped = true;
      p.what = "clip base 100% + clipped multiply";
    });
    stack("clip-base-50", DocumentFill.WHITE, function (d, p) {
      var base = layer(d, "base");
      ellipse(d, 8, 8, 56, 56, rgb(40, 60, 80), 0);
      base.opacity = 50;
      var c = layer(d, "clipped multiply");
      rect(d, 0, 0, 64, 64, rgb(128, 128, 128), 0);
      c.blendMode = BlendMode.MULTIPLY;
      c.grouped = true;
      p.what = "clip base 50% + clipped multiply";
    });
    stack("layer-mask", DocumentFill.WHITE, function (d, p) {
      var a = layer(d, "masked");
      rect(d, 0, 0, 64, 64, rgb(220, 40, 160), 0);
      d.selection.select([[10, 10], [54, 10], [54, 54], [10, 54]]);
      d.selection.feather(5);
      revealSelectionMask();
      d.selection.deselect();
      p.what = "full layer with a feathered rectangular layer mask";
    });
    stack("group-pass-through", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 32, 64, rgb(250, 200, 30), 0);
      var g = d.layerSets.add();
      g.name = "group";
      var a = g.artLayers.add();
      a.name = "multiply in group";
      d.activeLayer = a;
      rect(d, 16, 16, 48, 48, rgb(60, 140, 220), 0);
      a.blendMode = BlendMode.MULTIPLY;
      p.what = "pass-through group holding a multiply layer";
    });
    stack("group-isolated-50", DocumentFill.WHITE, function (d, p) {
      background(d);
      rect(d, 0, 0, 32, 64, rgb(250, 200, 30), 0);
      var g = d.layerSets.add();
      g.name = "group";
      g.opacity = 50;
      g.blendMode = BlendMode.NORMAL;
      var a = g.artLayers.add();
      d.activeLayer = a;
      rect(d, 16, 16, 48, 48, rgb(60, 140, 220), 0);
      var b = g.artLayers.add();
      d.activeLayer = b;
      rect(d, 24, 4, 60, 40, rgb(200, 30, 30), 0);
      p.what = "normal group at 50% holding two overlapping layers";
    });
  } finally {
    app.preferences.maximizeCompatibility = prevCompat;
  }
  return P.finish([]);
})();
