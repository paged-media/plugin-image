// TEXT layers, written by Photoshop itself: a line of type over white,
// dark and coloured backdrops, at 60 % opacity and in Multiply. Photoshop
// composites text in a gamma space ("Blend Text Colors Using Gamma"), so
// a plain normal blend of the stored text pixels misses its anti-aliased
// edges; the replay (psd_text_layers.rs) holds our gamma blend to
// Photoshop's merged composite.
(function () {
  var P = PagedProbe;
  var st = P.begin("text-layers");
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
  function text(d, contents, size, color) {
    var l = d.artLayers.add();
    l.kind = LayerKind.TEXT;
    var t = l.textItem;
    t.contents = contents;
    t.size = size;
    t.position = [4, 40];
    t.color = color;
    return l;
  }

  function stack(id, params, build) {
    var c = { id: id, params: { backdrop: [params.bg.rgb.red, params.bg.rgb.green, params.bg.rgb.blue] } };
    var d = null;
    try {
      d = app.documents.add(160, S, 72, id, NewDocumentMode.RGB, DocumentFill.WHITE);
      d.activeLayer = d.artLayers[d.artLayers.length - 1];
      d.selection.selectAll();
      d.selection.fill(params.bg);
      d.selection.deselect();
      build(d);
      var psd = new PhotoshopSaveOptions();
      psd.layers = true;
      psd.embedColorProfile = false;
      psd.alphaChannels = false;
      d.saveAs(new File(PAGED_STAGE + "/text-layers/" + id + ".psd"), psd, true, Extension.LOWERCASE);
      var png = new PNGSaveOptions();
      png.compression = 9;
      d.saveAs(new File(PAGED_STAGE + "/text-layers/" + id + ".png"), png, true, Extension.LOWERCASE);
      c.files = ["text-layers/" + id + ".psd", "text-layers/" + id + ".png"];
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

  function backdrop(r, g, b) {
    return rgb(r, g, b);
  }
  try {
    stack("black-on-white", { bg: backdrop(255, 255, 255) }, function (d) {
      text(d, "Ragged 0123", 22, rgb(0, 0, 0));
    });
    stack("white-on-dark", { bg: backdrop(30, 40, 60) }, function (d) {
      text(d, "Ragged 0123", 22, rgb(255, 255, 255));
    });
    stack("red-on-green", { bg: backdrop(40, 160, 60) }, function (d) {
      text(d, "Ragged 0123", 16, rgb(220, 30, 30));
    });
    stack("text-at-60", { bg: backdrop(240, 230, 210) }, function (d) {
      var l = text(d, "Ragged 0123", 22, rgb(20, 40, 120));
      l.opacity = 60;
    });
    stack("text-multiply", { bg: backdrop(240, 200, 120) }, function (d) {
      var l = text(d, "Ragged 0123", 22, rgb(60, 120, 200));
      l.blendMode = BlendMode.MULTIPLY;
    });
  } finally {
    app.preferences.maximizeCompatibility = prevCompat;
  }
  return P.finish([]);
})();
