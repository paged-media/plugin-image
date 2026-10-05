// Adjustment LAYERS, written by Photoshop itself: each case is the
// colour-sweep stimulus (reduced to 8 bits, as the layer import reads
// it) with one or two adjustment layers above it, saved as a layered PSD
// with REAL merged data plus a PNG of the same document. The replay
// (psd_adjustment_layers.rs) parses each PSD's adjustment blocks
// (`curv`, `levl`, `expA`, `nvrt`, `hue2`, ...), composites our stack
// and compares it with Photoshop's merged composite.
//
// "Maximize compatibility" is what makes Photoshop write the merged
// composite; the preference is set for the run and restored after.
(function () {
  var P = PagedProbe;
  var st = P.begin("adjustment-layers");
  var prevCompat = app.preferences.maximizeCompatibility;
  app.preferences.maximizeCompatibility = QueryStateType.ALWAYS;
  st.maximize_compatibility_was = String(prevCompat);
  var cid = P.cid;

  function adjLayer(cls, type) {
    var d = new ActionDescriptor();
    var r = new ActionReference();
    r.putClass(cid("AdjL"));
    d.putReference(cid("null"), r);
    var u = new ActionDescriptor();
    u.putObject(cid("Type"), cid(cls), type);
    d.putObject(cid("Usng"), cid("AdjL"), u);
    executeAction(cid("Mk  "), d, DialogModes.NO);
  }
  function chan(code) {
    var r = new ActionReference();
    r.putEnumerated(cid("Chnl"), cid("Chnl"), cid(code));
    return r;
  }
  // curves: [[channel, [[in, out], ...]], ...]
  function curves(spec) {
    var t = new ActionDescriptor();
    t.putEnumerated(cid("Prst"), cid("Prst"), cid("Cstm"));
    var adjs = new ActionList();
    var i, j;
    for (i = 0; i < spec.length; i++) {
      var c = new ActionDescriptor();
      c.putReference(cid("Chnl"), chan(spec[i][0]));
      var pts = new ActionList();
      for (j = 0; j < spec[i][1].length; j++) {
        var p = new ActionDescriptor();
        p.putDouble(cid("Hrzn"), spec[i][1][j][0]);
        p.putDouble(cid("Vrtc"), spec[i][1][j][1]);
        pts.putObject(cid("Pnt "), p);
      }
      c.putList(cid("Crv "), pts);
      adjs.putObject(cid("CrvA"), c);
    }
    t.putList(cid("Adjs"), adjs);
    adjLayer("Crvs", t);
  }
  // levels: [[channel, inBlack, inWhite, gamma, outBlack, outWhite], ...]
  function levels(spec) {
    var t = new ActionDescriptor();
    t.putEnumerated(cid("Prst"), cid("Prst"), cid("Cstm"));
    var adjs = new ActionList();
    var i;
    for (i = 0; i < spec.length; i++) {
      var s = spec[i];
      var c = new ActionDescriptor();
      c.putReference(cid("Chnl"), chan(s[0]));
      var inp = new ActionList();
      inp.putInteger(s[1]);
      inp.putInteger(s[2]);
      c.putList(cid("Inpt"), inp);
      c.putDouble(cid("Gmm "), s[3]);
      var out = new ActionList();
      out.putInteger(s[4]);
      out.putInteger(s[5]);
      c.putList(cid("Otpt"), out);
      adjs.putObject(cid("LvlA"), c);
    }
    t.putList(cid("Adjs"), adjs);
    adjLayer("Lvls", t);
  }
  function exposure(e, o, g) {
    var t = new ActionDescriptor();
    t.putDouble(cid("Exps"), e);
    t.putDouble(cid("Ofst"), o);
    t.putDouble(cid("gammaCorrection"), g);
    adjLayer("Exps", t);
  }
  // hueSat(colorize, [[rangeIndex (0 = master), h, s, l], ...])
  function hueSat(colorize, spec) {
    var t = new ActionDescriptor();
    t.putEnumerated(cid("Prst"), cid("Prst"), cid("Cstm"));
    t.putBoolean(cid("Clrz"), colorize);
    var adjs = new ActionList();
    var i;
    // Photoshop's default range bounds (begin ramp, begin sustain, end
    // sustain, end ramp) for reds .. magentas.
    var bounds = [
      [315, 345, 15, 45],
      [15, 45, 75, 105],
      [75, 105, 135, 165],
      [135, 165, 195, 225],
      [195, 225, 255, 285],
      [255, 285, 315, 345]
    ];
    for (i = 0; i < spec.length; i++) {
      var s = spec[i];
      var a = new ActionDescriptor();
      if (s[0] > 0) {
        var b = bounds[s[0] - 1];
        a.putInteger(cid("LclR"), s[0]);
        a.putInteger(cid("BgnR"), b[0]);
        a.putInteger(cid("BgnS"), b[1]);
        a.putInteger(cid("EndS"), b[2]);
        a.putInteger(cid("EndR"), b[3]);
      }
      a.putInteger(cid("H   "), s[1]);
      a.putInteger(cid("Strt"), s[2]);
      a.putInteger(cid("Lght"), s[3]);
      adjs.putObject(cid("Hst2"), a);
    }
    t.putList(cid("Adjs"), adjs);
    adjLayer("HStr", t);
  }
  function photoFilter(r, g, b, density, keep) {
    var t = new ActionDescriptor();
    var c = new ActionDescriptor();
    c.putDouble(cid("Rd  "), r);
    c.putDouble(cid("Grn "), g);
    c.putDouble(cid("Bl  "), b);
    t.putObject(cid("Clr "), cid("RGBC"), c);
    t.putInteger(cid("Dnst"), density);
    t.putBoolean(cid("PrsL"), keep);
    adjLayer("photoFilter", t);
  }
  function brightness(b, c, legacy) {
    var t = new ActionDescriptor();
    t.putInteger(cid("Brgh"), b);
    t.putInteger(cid("Cntr"), c);
    t.putBoolean(cid("useLegacy"), legacy);
    adjLayer("BrgC", t);
  }

  function stack(id, params, build) {
    var c = { id: id, params: params };
    var d = null;
    try {
      d = P.open("stimuli/colour-sweep.png");
      d.bitsPerChannel = BitsPerChannelType.EIGHT;
      build(d);
      var psd = new PhotoshopSaveOptions();
      psd.layers = true;
      psd.embedColorProfile = false;
      psd.alphaChannels = false;
      d.saveAs(new File(PAGED_STAGE + "/adjustment-layers/" + id + ".psd"), psd, true, Extension.LOWERCASE);
      var png = new PNGSaveOptions();
      png.compression = 9;
      d.saveAs(new File(PAGED_STAGE + "/adjustment-layers/" + id + ".png"), png, true, Extension.LOWERCASE);
      c.files = ["adjustment-layers/" + id + ".psd", "adjustment-layers/" + id + ".png"];
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

  var S_CURVE = [[0, 0], [64, 40], [128, 150], [192, 220], [255, 255]];
  try {
    stack("curves-composite", { composite: S_CURVE }, function (d) {
      curves([["Cmps", S_CURVE]]);
    });
    stack("curves-channels", {
      composite: [[0, 0], [128, 110], [255, 255]],
      red: [[0, 0], [128, 180], [255, 255]],
      blue: [[0, 30], [255, 220]]
    }, function (d) {
      curves([
        ["Cmps", [[0, 0], [128, 110], [255, 255]]],
        ["Rd  ", [[0, 0], [128, 180], [255, 255]]],
        ["Bl  ", [[0, 30], [255, 220]]]
      ]);
    });
    stack("levels-composite", { composite: [20, 230, 1.4, 10, 245] }, function (d) {
      levels([["Cmps", 20, 230, 1.4, 10, 245]]);
    });
    stack("levels-channels", {
      composite: [10, 240, 1.2, 0, 255],
      red: [0, 200, 0.8, 0, 255],
      green: [30, 255, 1.0, 20, 255]
    }, function (d) {
      levels([
        ["Cmps", 10, 240, 1.2, 0, 255],
        ["Rd  ", 0, 200, 0.8, 0, 255],
        ["Grn ", 30, 255, 1.0, 20, 255]
      ]);
    });
    stack("exposure", { exposure: 0.7, offset: 0, gamma: 1 }, function (d) {
      exposure(0.7, 0, 1);
    });
    stack("exposure-offset-gamma", { exposure: 0.4, offset: -0.03, gamma: 1.2 }, function (d) {
      exposure(0.4, -0.03, 1.2);
    });
    stack("invert", {}, function (d) {
      adjLayer("Invr", new ActionDescriptor());
    });
    stack("hue-sat-master", { master: [40, 30, 0] }, function (d) {
      hueSat(false, [[0, 40, 30, 0]]);
    });
    stack("hue-sat-reds", { master: [0, 0, 0], reds: [-30, 20, 0] }, function (d) {
      hueSat(false, [[0, 0, 0, 0], [1, -30, 20, 0]]);
    });
    stack("hue-sat-colorize", { colorize: [200, 50, 0] }, function (d) {
      hueSat(true, [[0, 200, 50, 0]]);
    });
    stack("curves-at-60", { composite: S_CURVE, opacity: 60 }, function (d) {
      curves([["Cmps", S_CURVE]]);
      d.activeLayer.opacity = 60;
    });
    stack("invert-clipped", { what: "an invert clipped to a half-canvas layer" }, function (d) {
      var l = d.artLayers.add();
      l.name = "half";
      d.selection.select([[0, 0], [128, 0], [128, 64], [0, 64]]);
      var col = new SolidColor();
      col.rgb.red = 220;
      col.rgb.green = 120;
      col.rgb.blue = 40;
      d.selection.fill(col);
      d.selection.deselect();
      adjLayer("Invr", new ActionDescriptor());
      d.activeLayer.grouped = true;
    });
    stack("photo-filter", { color: [236, 138, 0], density: 50, preserve_luminosity: true }, function (d) {
      photoFilter(236, 138, 0, 50, true);
    });
    stack("brightness-contrast", { brightness: 30, contrast: 40, legacy: false }, function (d) {
      brightness(30, 40, false);
    });
  } finally {
    app.preferences.maximizeCompatibility = prevCompat;
  }
  return P.finish([]);
})();
