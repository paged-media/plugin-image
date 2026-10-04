// Filters applied to `colour-sweep` (16-bit RGB). Radii in pixels.
(function () {
  var P = PagedProbe;
  P.begin("filters");
  var src = P.open("stimuli/colour-sweep.png");
  function L(d) {
    return d.activeLayer;
  }
  P.run("identity", src, {}, function (d) {});
  P.run("gaussian-r2", src, { radius: 2 }, function (d) {
    L(d).applyGaussianBlur(2);
  });
  P.run("gaussian-r5", src, { radius: 5 }, function (d) {
    L(d).applyGaussianBlur(5);
  });
  P.run("unsharp-100-r1.5", src, { amount: 100, radius: 1.5, threshold: 0 }, function (d) {
    L(d).applyUnSharpMask(100, 1.5, 0);
  });
  P.run("motion-30-r10", src, { angle: 30, distance: 10 }, function (d) {
    L(d).applyMotionBlur(30, 10);
  });
  P.run("emboss", src, { angle: 135, height: 3, amount: 100 }, function (d) {
    P.exec("Embs", [["Angl", "int", 135], ["Hght", "int", 3], ["Amnt", "int", 100]]);
  });
  P.run("find-edges", src, {}, function (d) {
    P.exec("FndE", []);
  });
  P.run("mosaic-8", src, { cell: 8 }, function (d) {
    P.exec("Msc ", [["ClSz", "unit", ["#Pxl", 8]]]);
  });
  P.run("median-r1", src, { radius: 1 }, function (d) {
    L(d).applyMedianNoise(1);
  });
  P.run("offset-20-10-wrap", src, { h: 20, v: 10, undefined_areas: "wrap" }, function (d) {
    L(d).applyOffset(20, 10, OffsetUndefinedAreas.WRAPAROUND);
  });
  return P.finish([src]);
})();
