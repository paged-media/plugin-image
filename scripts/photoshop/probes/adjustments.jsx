// The shipped adjustments, each applied to `colour-sweep` (16-bit RGB).
// Parameters are Photoshop's own units; `params` records them and the
// replay maps them onto the engine's.
(function () {
  var P = PagedProbe;
  P.begin("adjustments");
  var src = P.open("stimuli/colour-sweep.png");
  function L(d) {
    return d.activeLayer;
  }
  function bc(b, c, legacy) {
    P.exec("BrgC", [
      ["Brgh", "int", b],
      ["Cntr", "int", c],
      ["useLegacy", "bool", legacy]
    ]);
  }
  P.run("identity", src, {}, function (d) {});
  P.run("invert", src, {}, function (d) {
    L(d).invert();
  });
  P.run("posterize-4", src, { levels: 4 }, function (d) {
    L(d).posterize(4);
  });
  P.run("threshold-128", src, { level: 128 }, function (d) {
    L(d).threshold(128);
  });
  P.run("levels", src, { in_black: 20, in_white: 230, gamma: 1.4, out_black: 10, out_white: 245 }, function (d) {
    L(d).adjustLevels(20, 230, 1.4, 10, 245);
  });
  P.run("curves", src, { points: [[0, 0], [64, 40], [128, 150], [192, 220], [255, 255]] }, function (d) {
    L(d).adjustCurves([[0, 0], [64, 40], [128, 150], [192, 220], [255, 255]]);
  });
  P.run("exposure+1", src, { exposure: 1, offset: 0, gamma: 1 }, function (d) {
    P.exec("Exps", [["Exps", "dbl", 1.0], ["Ofst", "dbl", 0.0], ["gammaCorrection", "dbl", 1.0]]);
  });
  P.run("exposure-0.5", src, { exposure: -0.5, offset: 0, gamma: 1 }, function (d) {
    P.exec("Exps", [["Exps", "dbl", -0.5], ["Ofst", "dbl", 0.0], ["gammaCorrection", "dbl", 1.0]]);
  });
  P.run("brightness-contrast-legacy", src, { brightness: 30, contrast: 40, legacy: true }, function (d) {
    bc(30, 40, true);
  });
  P.run("brightness-contrast", src, { brightness: 30, contrast: 40, legacy: false }, function (d) {
    bc(30, 40, false);
  });
  P.run("hue-saturation", src, { hue: 40, saturation: 30, lightness: 0 }, function (d) {
    var list = new ActionList();
    var a = new ActionDescriptor();
    a.putInteger(P.cid("H   "), 40);
    a.putInteger(P.cid("Strt"), 30);
    a.putInteger(P.cid("Lght"), 0);
    list.putObject(P.cid("Hst2"), a);
    P.exec("HStr", [["Clrz", "bool", false], ["Adjs", "list", list]]);
  });
  P.run("hue-only", src, { hue: 40, saturation: 0, lightness: 0 }, function (d) {
    var list = new ActionList();
    var a = new ActionDescriptor();
    a.putInteger(P.cid("H   "), 40);
    a.putInteger(P.cid("Strt"), 0);
    a.putInteger(P.cid("Lght"), 0);
    list.putObject(P.cid("Hst2"), a);
    P.exec("HStr", [["Clrz", "bool", false], ["Adjs", "list", list]]);
  });
  P.run("vibrance", src, { vibrance: 50, saturation: 0 }, function (d) {
    P.exec("vibrance", [["vibrance", "int", 50], ["Strt", "int", 0]]);
  });
  P.run("color-balance", src, { shadows: [0, 0, 0], midtones: [30, -20, 10], highlights: [0, 0, 0], preserve_luminosity: true }, function (d) {
    L(d).adjustColorBalance([0, 0, 0], [30, -20, 10], [0, 0, 0], true);
  });
  P.run("channel-mixer", src, { red: [80, 30, -10, 0], green: [0, 100, 0, 0], blue: [0, 20, 80, 0] }, function (d) {
    L(d).mixChannels([[80, 30, -10, 0], [0, 100, 0, 0], [0, 20, 80, 0]], false);
  });
  P.run("photo-filter", src, { color: [236, 138, 0], density: 50, preserve_luminosity: true }, function (d) {
    var c = new SolidColor();
    c.rgb.red = 236;
    c.rgb.green = 138;
    c.rgb.blue = 0;
    L(d).photoFilter(c, 50, true);
  });
  P.run("black-white", src, { weights: [40, 60, 40, 60, 20, 80] }, function (d) {
    P.exec("BanW", [
      ["Rd  ", "int", 40],
      ["Yllw", "int", 60],
      ["Grn ", "int", 40],
      ["Cyn ", "int", 60],
      ["Bl  ", "int", 20],
      ["Mgnt", "int", 80],
      ["useTint", "bool", false]
    ]);
  });
  return P.finish([src]);
})();
