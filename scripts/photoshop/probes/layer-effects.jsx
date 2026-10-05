// Layer EFFECTS, written by Photoshop itself: small RGB documents with a
// Color Overlay in each mode the corpus uses, on a masked layer, on a
// clip base, under layer opacity and on soft edges; plus a layer whose
// Stroke is switched OFF (it must import as if it had no effects) and one
// whose Stroke is on (it must be refused). Each is saved as a layered PSD
// with REAL merged data plus a PNG of the same document; the replay is
// psd_layer_effects.rs.
(function () {
  var P = PagedProbe;
  var st = P.begin("layer-effects");
  var S = 64;
  var cid = P.cid;
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
  function rect(d, x0, y0, x1, y1, color) {
    d.selection.select([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    d.selection.fill(color);
    d.selection.deselect();
  }
  function layer(d, name) {
    var l = d.artLayers.add();
    l.name = name;
    return l;
  }
  function rgbDesc(r, g, b) {
    var c = new ActionDescriptor();
    c.putDouble(cid("Rd  "), r);
    c.putDouble(cid("Grn "), g);
    c.putDouble(cid("Bl  "), b);
    return c;
  }
  // effects on the active layer: { overlay: [mode, opacity, r, g, b],
  // stroke: [enabled, size] }
  function effects(spec) {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putProperty(cid("Prpr"), cid("Lefx"));
    ref.putEnumerated(cid("Lyr "), cid("Ordn"), cid("Trgt"));
    desc.putReference(cid("null"), ref);
    var fx = new ActionDescriptor();
    fx.putUnitDouble(cid("Scl "), cid("#Prc"), 100);
    if (spec.overlay) {
      var o = spec.overlay;
      var so = new ActionDescriptor();
      so.putBoolean(cid("enab"), true);
      so.putEnumerated(cid("Md  "), cid("BlnM"), cid(o[0]));
      so.putUnitDouble(cid("Opct"), cid("#Prc"), o[1]);
      so.putObject(cid("Clr "), cid("RGBC"), rgbDesc(o[2], o[3], o[4]));
      fx.putObject(cid("SoFi"), cid("SoFi"), so);
    }
    if (spec.stroke) {
      var s = spec.stroke;
      var fr = new ActionDescriptor();
      fr.putBoolean(cid("enab"), s[0]);
      fr.putEnumerated(cid("Styl"), cid("FStl"), cid("OutF"));
      fr.putEnumerated(cid("PntT"), cid("FrFl"), cid("SClr"));
      fr.putEnumerated(cid("Md  "), cid("BlnM"), cid("Nrml"));
      fr.putUnitDouble(cid("Opct"), cid("#Prc"), 100);
      fr.putUnitDouble(cid("Sz  "), cid("#Pxl"), s[1]);
      fr.putObject(cid("Clr "), cid("RGBC"), rgbDesc(20, 20, 20));
      fx.putObject(cid("FrFX"), cid("FrFX"), fr);
    }
    desc.putObject(cid("T   "), cid("Lefx"), fx);
    executeAction(cid("setd"), desc, DialogModes.NO);
  }
  function revealSelectionMask() {
    var desc = new ActionDescriptor();
    desc.putClass(cid("Nw  "), cid("Chnl"));
    var ref = new ActionReference();
    ref.putEnumerated(cid("Chnl"), cid("Chnl"), cid("Msk "));
    desc.putReference(cid("At  "), ref);
    desc.putEnumerated(cid("Usng"), cid("UsrM"), cid("RvlS"));
    executeAction(cid("Mk  "), desc, DialogModes.NO);
  }

  function stack(id, params, build) {
    var c = { id: id, params: params };
    var d = null;
    try {
      d = app.documents.add(S, S, 72, id, NewDocumentMode.RGB, DocumentFill.WHITE);
      d.activeLayer = d.artLayers[d.artLayers.length - 1];
      rect(d, 0, 0, 32, 64, rgb(240, 190, 40));
      build(d);
      var psd = new PhotoshopSaveOptions();
      psd.layers = true;
      psd.embedColorProfile = false;
      psd.alphaChannels = false;
      d.saveAs(new File(PAGED_STAGE + "/layer-effects/" + id + ".psd"), psd, true, Extension.LOWERCASE);
      var png = new PNGSaveOptions();
      png.compression = 9;
      d.saveAs(new File(PAGED_STAGE + "/layer-effects/" + id + ".png"), png, true, Extension.LOWERCASE);
      c.files = ["layer-effects/" + id + ".psd", "layer-effects/" + id + ".png"];
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

  function overlayCase(id, mode, opacity, feather) {
    stack(id, { mode: mode, opacity: opacity, colour: [40, 90, 200], feathered: !!feather }, function (d) {
      layer(d, "shape");
      ellipse(d, 8, 8, 56, 56, rgb(200, 60, 40), feather || 0);
      effects({ overlay: [mode, opacity, 40, 90, 200] });
    });
  }

  try {
    overlayCase("overlay-normal-100", "Nrml", 100);
    overlayCase("overlay-normal-90", "Nrml", 90);
    overlayCase("overlay-multiply-50", "Mltp", 50);
    overlayCase("overlay-linear-burn-100", "linearBurn", 100);
    overlayCase("overlay-overlay-45", "Ovrl", 45);
    overlayCase("overlay-soft-edge", "Nrml", 100, 5);
    stack("overlay-masked", { mode: "Nrml", opacity: 70, mask: "feathered rectangle" }, function (d) {
      layer(d, "shape");
      rect(d, 0, 0, 64, 64, rgb(200, 60, 40));
      d.selection.select([[10, 10], [54, 10], [54, 54], [10, 54]]);
      d.selection.feather(4);
      revealSelectionMask();
      d.selection.deselect();
      effects({ overlay: ["Nrml", 70, 40, 90, 200] });
    });
    stack("overlay-layer-opacity-60", { mode: "Mltp", opacity: 100, layer_opacity: 60 }, function (d) {
      var l = layer(d, "shape");
      ellipse(d, 8, 8, 56, 56, rgb(200, 60, 40), 0);
      effects({ overlay: ["Mltp", 100, 40, 90, 200] });
      l.opacity = 60;
    });
    stack("overlay-clip-base", { mode: "Nrml", opacity: 100, clipped: "multiply grey above the base" }, function (d) {
      layer(d, "base");
      ellipse(d, 8, 8, 56, 56, rgb(200, 60, 40), 0);
      effects({ overlay: ["Nrml", 100, 40, 90, 200] });
      var c = layer(d, "clipped multiply");
      rect(d, 0, 32, 64, 64, rgb(128, 128, 128));
      c.blendMode = BlendMode.MULTIPLY;
      c.grouped = true;
    });
    stack("stroke-disabled", { stroke: "off" }, function (d) {
      layer(d, "shape");
      ellipse(d, 8, 8, 56, 56, rgb(200, 60, 40), 0);
      effects({ stroke: [false, 3] });
    });
    stack("stroke-enabled", { stroke: "on, 3 px outside" }, function (d) {
      layer(d, "shape");
      ellipse(d, 8, 8, 56, 56, rgb(200, 60, 40), 0);
      effects({ stroke: [true, 3] });
    });
  } finally {
    app.preferences.maximizeCompatibility = prevCompat;
  }
  return P.finish([]);
})();
