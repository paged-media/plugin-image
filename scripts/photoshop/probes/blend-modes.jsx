// The 26 blend modes at 100 % and 50 % layer opacity: `blend-top` as a
// layer over `blend-bottom`, flattened. Recorded case ids are
// `<mode>@<opacity>`; mode names are the engine's compose.* names.
(function () {
  var P = PagedProbe;
  P.begin("blend-modes");
  var bottom = P.open("stimuli/blend-bottom.png");
  var top = P.open("stimuli/blend-top.png");
  var MODES = [
    ["normal", BlendMode.NORMAL],
    ["multiply", BlendMode.MULTIPLY],
    ["screen", BlendMode.SCREEN],
    ["overlay", BlendMode.OVERLAY],
    ["darken", BlendMode.DARKEN],
    ["lighten", BlendMode.LIGHTEN],
    ["color_dodge", BlendMode.COLORDODGE],
    ["color_burn", BlendMode.COLORBURN],
    ["hard_light", BlendMode.HARDLIGHT],
    ["soft_light", BlendMode.SOFTLIGHT],
    ["difference", BlendMode.DIFFERENCE],
    ["exclusion", BlendMode.EXCLUSION],
    ["hue", BlendMode.HUE],
    ["saturation", BlendMode.SATURATION],
    ["color", BlendMode.COLORBLEND],
    ["luminosity", BlendMode.LUMINOSITY],
    ["linear_burn", BlendMode.LINEARBURN],
    ["linear_dodge", BlendMode.LINEARDODGE],
    ["darker_color", BlendMode.DARKERCOLOR],
    ["lighter_color", BlendMode.LIGHTERCOLOR],
    ["vivid_light", BlendMode.VIVIDLIGHT],
    ["linear_light", BlendMode.LINEARLIGHT],
    ["pin_light", BlendMode.PINLIGHT],
    ["hard_mix", BlendMode.HARDMIX],
    ["subtract", BlendMode.SUBTRACT],
    ["divide", BlendMode.DIVIDE]
  ];
  var OPACITIES = [100, 50];
  // The two-layer base every case duplicates.
  app.activeDocument = top;
  top.activeLayer.duplicate(bottom, ElementPlacement.PLACEATBEGINNING);
  app.activeDocument = bottom;
  P.run("identity-bottom", bottom, {}, function (d) {
    d.layers[0].visible = false;
  });
  var i, j;
  for (i = 0; i < MODES.length; i++) {
    for (j = 0; j < OPACITIES.length; j++) {
      (function (m, o) {
        P.run(m[0] + "@" + o, bottom, { mode: m[0], opacity: o }, function (d) {
          var l = d.layers[0];
          l.blendMode = m[1];
          l.opacity = o;
        });
      })(MODES[i], OPACITIES[j]);
    }
  }
  return P.finish([top, bottom]);
})();
