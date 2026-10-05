// Layered RGB PSDs whose layers carry VECTOR MASKS, written by Photoshop
// itself with REAL merged data, plus a PNG of the same document. The
// replay (psd_vector_masks_photoshop.rs) parses each PSD, draws the
// vector masks with our rasterizer, flattens our layer stack and holds
// it to Photoshop's merged composite -- and holds our rasterization to
// the render Photoshop caches in the layer's mask channel.
//
// What each case pins (the findings are in image-psd/src/vector_mask.rs
// and image-js/src/psd_vector_mask.rs):
//   * anti-aliasing: fractional rectangle edges, a 1/16-pixel edge ramp,
//     diagonals, a curve, an open curved subpath;
//   * the path operations Photoshop offers per component (combine,
//     subtract, intersect, exclude), a FIRST component that subtracts,
//     and per-SAMPLE booleans (an intersect of two edges 1/2 px apart, a
//     shared anti-aliased diagonal);
//   * a user mask beside a vector mask, each one disabled in turn, and a
//     disabled vector mask alone;
//   * a layer smaller than the canvas;
//   * the NON-ZERO fill rule, a hole made by a second subpath of the
//     SAME component, and the invert flag. The scripting DOM cannot set
//     any of these, so the probe writes the case with the DOM, patches
//     the one field in the saved bytes, has Photoshop OPEN the patched
//     file and SAVE it again: the recorded PSD and its composite are
//     Photoshop's own;
//   * shape layers (a fill layer with a vector mask; a live ellipse),
//     whose stored pixels already are the shape;
//   * two cases the import still refuses: a vector mask with a feather,
//     and a group with a vector mask.
//
// "Maximize compatibility" is set for the run and restored after.
(function () {
  var P = PagedProbe;
  var st = P.begin("vector-masks");
  var S = 64;
  var DIR = PAGED_STAGE + "/vector-masks/";
  new Folder(DIR).create();
  var prevCompat = app.preferences.maximizeCompatibility;
  app.preferences.maximizeCompatibility = QueryStateType.ALWAYS;
  st.maximize_compatibility_was = String(prevCompat);

  var ADD = ShapeOperation.SHAPEADD;
  var SUB = ShapeOperation.SHAPESUBTRACT;
  var INT = ShapeOperation.SHAPEINTERSECT;
  var XOR = ShapeOperation.SHAPEXOR;

  function rgb(r, g, b) {
    var c = new SolidColor();
    c.rgb.red = r;
    c.rgb.green = g;
    c.rgb.blue = b;
    return c;
  }
  function corner(x, y) {
    var p = new PathPointInfo();
    p.kind = PointKind.CORNERPOINT;
    p.anchor = [x, y];
    p.leftDirection = [x, y];
    p.rightDirection = [x, y];
    return p;
  }
  // DOM handles: rightDirection is the one the path ARRIVES with in the
  // saved record, leftDirection the one it LEAVES with.
  function smooth(ax, ay, inX, inY, outX, outY) {
    var p = new PathPointInfo();
    p.kind = PointKind.SMOOTHPOINT;
    p.anchor = [ax, ay];
    p.rightDirection = [inX, inY];
    p.leftDirection = [outX, outY];
    return p;
  }
  function sub(points, op, closed) {
    var s = new SubPathInfo();
    s.closed = closed !== false;
    s.operation = op;
    s.entireSubPath = points;
    return s;
  }
  function rect(x0, y0, x1, y1, op) {
    return sub([corner(x0, y0), corner(x1, y0), corner(x1, y1), corner(x0, y1)], op);
  }
  function circle(cx, cy, r, op) {
    var k = 0.5522847498 * r;
    return sub(
      [
        smooth(cx + r, cy, cx + r, cy - k, cx + r, cy + k),
        smooth(cx, cy + r, cx + k, cy + r, cx - k, cy + r),
        smooth(cx - r, cy, cx - r, cy + k, cx - r, cy - k),
        smooth(cx, cy - r, cx - k, cy - r, cx + k, cy - r)
      ],
      op
    );
  }

  // -- ActionManager ----------------------------------------------------
  function vectorMaskFromPath(d, name) {
    d.pathItems.getByName(name).select();
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putClass(P.cid("Path"));
    desc.putReference(P.cid("null"), ref);
    var at = new ActionReference();
    at.putEnumerated(P.cid("Path"), P.cid("Path"), P.cid("vectorMask"));
    desc.putReference(P.cid("At  "), at);
    var using = new ActionReference();
    using.putEnumerated(P.cid("Path"), P.cid("Ordn"), P.cid("Trgt"));
    desc.putReference(P.cid("Usng"), using);
    executeAction(P.cid("Mk  "), desc, DialogModes.NO);
  }
  function addVectorMask(d, subpaths) {
    d.pathItems.add("vm", subpaths);
    vectorMaskFromPath(d, "vm");
    d.pathItems.getByName("vm").remove();
  }
  function setLayer(entries) {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putEnumerated(P.cid("Lyr "), P.cid("Ordn"), P.cid("Trgt"));
    desc.putReference(P.cid("null"), ref);
    var to = new ActionDescriptor();
    var i, e;
    for (i = 0; i < entries.length; i++) {
      e = entries[i];
      if (e[1] === "bool") to.putBoolean(P.cid(e[0]), e[2]);
      else to.putUnitDouble(P.cid(e[0]), P.cid(e[1]), e[2]);
    }
    desc.putObject(P.cid("T   "), P.cid("Lyr "), to);
    executeAction(P.cid("setd"), desc, DialogModes.NO);
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
  function userMask(d, x0, y0, x1, y1, feather) {
    d.selection.select([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    d.selection.feather(feather);
    revealSelectionMask();
    d.selection.deselect();
  }
  function contentLayer(shape) {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putClass(P.cid("contentLayer"));
    desc.putReference(P.cid("null"), ref);
    var cl = new ActionDescriptor();
    var fill = new ActionDescriptor();
    var c = new ActionDescriptor();
    c.putDouble(P.cid("Rd  "), 220);
    c.putDouble(P.cid("Grn "), 40);
    c.putDouble(P.cid("Bl  "), 160);
    fill.putObject(P.cid("Clr "), P.cid("RGBC"), c);
    cl.putObject(P.cid("Type"), P.cid("solidColorLayer"), fill);
    if (shape) {
      var sh = new ActionDescriptor();
      sh.putUnitDouble(P.cid("Top "), P.cid("#Pxl"), shape[1]);
      sh.putUnitDouble(P.cid("Left"), P.cid("#Pxl"), shape[0]);
      sh.putUnitDouble(P.cid("Btom"), P.cid("#Pxl"), shape[3]);
      sh.putUnitDouble(P.cid("Rght"), P.cid("#Pxl"), shape[2]);
      cl.putObject(P.cid("Shp "), P.cid("Elps"), sh);
    }
    desc.putObject(P.cid("Usng"), P.cid("contentLayer"), cl);
    executeAction(P.cid("Mk  "), desc, DialogModes.NO);
  }

  // -- documents ----------------------------------------------------------
  function fullLayer(d) {
    var l = d.artLayers.add();
    l.name = "masked";
    d.selection.selectAll();
    d.selection.fill(rgb(220, 40, 160));
    d.selection.deselect();
    return l;
  }
  function masked(subpaths) {
    return function (d) {
      fullLayer(d);
      addVectorMask(d, subpaths);
    };
  }
  function savePsd(d, path) {
    var o = new PhotoshopSaveOptions();
    o.layers = true;
    o.embedColorProfile = false;
    o.alphaChannels = false;
    d.saveAs(new File(path), o, true, Extension.LOWERCASE);
  }
  function savePng(d, path) {
    var o = new PNGSaveOptions();
    o.compression = 9;
    d.saveAs(new File(path), o, true, Extension.LOWERCASE);
  }

  // Binary patching of a saved PSD's FIRST vmsk block (ExtendScript reads
  // and writes a BINARY file as a string of 0-255 char codes).
  function patchVmsk(path, edit) {
    var f = new File(path);
    f.encoding = "BINARY";
    if (!f.open("r")) throw new Error("cannot read " + path);
    var s = f.read();
    f.close();
    var at = s.indexOf("8BIMvmsk");
    if (at < 0) throw new Error("no vmsk block in " + path);
    var len =
      s.charCodeAt(at + 8) * 16777216 +
      s.charCodeAt(at + 9) * 65536 +
      s.charCodeAt(at + 10) * 256 +
      s.charCodeAt(at + 11);
    var payload = at + 12;
    var bytes = [];
    var i;
    for (i = 0; i < len; i++) bytes.push(s.charCodeAt(payload + i));
    edit(bytes);
    var out = s.substring(0, payload);
    for (i = 0; i < len; i++) out += String.fromCharCode(bytes[i]);
    out += s.substring(payload + len);
    f.encoding = "BINARY";
    if (!f.open("w")) throw new Error("cannot write " + path);
    f.write(out);
    f.close();
  }
  // Offsets (into the vmsk payload) of the subpath LENGTH records.
  function lengthRecords(bytes) {
    var out = [];
    var r, sel;
    for (r = 8; r + 26 <= bytes.length; r += 26) {
      sel = bytes[r] * 256 + bytes[r + 1];
      if (sel === 0 || sel === 3) out.push(r);
    }
    return out;
  }
  function put16(bytes, at, v) {
    bytes[at] = (v >> 8) & 255;
    bytes[at + 1] = v & 255;
  }

  function record(id, params, build, patch) {
    var c = { id: id, params: params };
    var d = null;
    try {
      d = app.documents.add(S, S, 72, id, NewDocumentMode.RGB, DocumentFill.WHITE);
      build(d);
      if (patch) {
        var src = DIR + id + ".unpatched.psd";
        savePsd(d, src);
        d.close(SaveOptions.DONOTSAVECHANGES);
        d = null;
        patchVmsk(src, patch);
        d = app.open(new File(src));
      }
      savePsd(d, DIR + id + ".psd");
      savePng(d, DIR + id + ".png");
      c.files = ["vector-masks/" + id + ".psd", "vector-masks/" + id + ".png"];
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
    var ramp = [];
    var i;
    for (i = 0; i < 16; i++) ramp.push(rect(4 * i + 1 + i / 16, 8 + i / 16, 4 * i + 3, 56 + i / 16, ADD));
    var star = [corner(32, 4), corner(49, 58), corner(4, 24), corner(60, 24), corner(15, 58)];

    record("rect-fractional", { what: "rectangle with edges at .25/.5/.75/.3" }, masked([rect(10.25, 10.5, 53.75, 40.3, ADD)]));
    record("edge-ramp", { what: "16 rectangles, edges at every 1/16 px" }, masked(ramp));
    record("triangle", { what: "diagonal edges" }, masked([sub([corner(4, 4), corner(60, 20), corner(10, 60)], ADD)]));
    record("star-even-odd", { what: "self-intersecting star, fill rule 1" }, masked([sub(star, ADD)]));
    record("circle", { what: "four smooth Bezier knots" }, masked([circle(32.4, 31.7, 26.3, ADD)]));
    record(
      "open-curve",
      { what: "OPEN curved subpath: fills as if a straight line closed it" },
      masked([
        sub([smooth(8, 50, 8, 50, 8, 10), smooth(56, 50, 56, 10, 56, 50), corner(40, 58)], ADD, false)
      ])
    );
    record("subtract", { what: "combine, then subtract" }, masked([rect(8, 8, 40, 40, ADD), rect(24, 24, 56, 56, SUB)]));
    record("intersect", { what: "combine, then intersect" }, masked([rect(8, 8, 40, 40, ADD), rect(24, 24, 56, 56, INT)]));
    record("exclude", { what: "combine, then exclude" }, masked([rect(8, 8, 40, 40, ADD), rect(24, 24, 56, 56, XOR)]));
    record("first-subtract", { what: "a first component that subtracts starts from a full mask" }, masked([rect(8, 8, 40, 40, SUB), rect(24, 24, 56, 56, ADD)]));
    record(
      "add-subtract-add",
      { what: "three components in order" },
      masked([rect(4, 4, 60, 60, ADD), rect(16, 16, 48, 48, SUB), rect(24, 24, 40, 40, ADD)])
    );
    record(
      "per-sample-intersect",
      { what: "intersect of edges at 10.25 and 10.75: the samples between them" },
      masked([rect(10.25, 10, 40, 50, ADD), rect(2, 10, 10.75, 50, INT)])
    );
    record(
      "per-sample-seam",
      { what: "two triangles sharing an anti-aliased diagonal: no seam" },
      masked([sub([corner(0, 0), corner(64, 0), corner(0, 64)], ADD), sub([corner(64, 0), corner(64, 64), corner(0, 64)], ADD)])
    );
    record("user-and-vector", { what: "feathered user mask x diagonal vector mask" }, function (d) {
      fullLayer(d);
      userMask(d, 4, 18, 60, 46, 5);
      addVectorMask(d, [sub([corner(20.3, 2), corner(46.6, 9.5), corner(38.2, 62)], ADD)]);
    });
    record("user-disabled", { what: "user mask disabled, vector mask applied" }, function (d) {
      fullLayer(d);
      userMask(d, 4, 18, 60, 46, 5);
      addVectorMask(d, [sub([corner(20.3, 2), corner(46.6, 9.5), corner(38.2, 62)], ADD)]);
      setLayer([["userMaskEnabled", "bool", false]]);
    });
    record("vector-disabled-with-user", { what: "vector mask disabled, user mask applied" }, function (d) {
      fullLayer(d);
      userMask(d, 4, 18, 60, 46, 5);
      addVectorMask(d, [sub([corner(20.3, 2), corner(46.6, 9.5), corner(38.2, 62)], ADD)]);
      setLayer([["vectorMaskEnabled", "bool", false]]);
    });
    record("vector-disabled", { what: "a disabled vector mask alone: kept, not applied" }, function (d) {
      fullLayer(d);
      addVectorMask(d, [rect(20, 4, 44, 60, ADD)]);
      setLayer([["vectorMaskEnabled", "bool", false]]);
    });
    record("partial-layer", { what: "a layer smaller than the canvas, mask reaching past it" }, function (d) {
      d.artLayers.add();
      d.selection.select([[8, 8], [40, 8], [40, 40], [8, 40]]);
      d.selection.fill(rgb(220, 40, 160));
      d.selection.deselect();
      addVectorMask(d, [sub([corner(20.5, 4), corner(60, 12.25), corner(30, 50.5)], ADD)]);
    });
    record(
      "star-non-zero",
      { what: "the star with its component's fill rule patched to 2 (non-zero)" },
      masked([sub(star, ADD)]),
      function (b) {
        put16(b, lengthRecords(b)[0] + 6, 2);
      }
    );
    record(
      "component-hole",
      { what: "inner rectangle patched to CONTINUE the outer one's component (0xFFFF): an even-odd hole" },
      masked([rect(8, 8, 56, 56, ADD), rect(20.5, 20.5, 43.5, 43.5, ADD)]),
      function (b) {
        var r = lengthRecords(b)[1];
        put16(b, r + 4, 65535);
        put16(b, r + 6, 0);
        put16(b, r + 14, 0);
      }
    );
    record(
      "inverted",
      { what: "vector mask flags patched to 1 (invert)" },
      masked([sub([corner(4, 4), corner(60, 20), corner(10, 60)], ADD)]),
      function (b) {
        b[7] = b[7] | 1;
      }
    );
    record("shape-fill", { what: "a solid-colour fill layer with a vector mask (shape layer)" }, function (d) {
      d.pathItems.add("vm", [sub([corner(4.3, 6.2), corner(58.6, 12.4), corner(30.25, 59.7)], ADD)]);
      d.pathItems.getByName("vm").select();
      contentLayer(null);
      d.pathItems.getByName("vm").remove();
    });
    record("shape-ellipse", { what: "a live ellipse shape layer" }, function (d) {
      contentLayer([6.5, 8.25, 57.5, 50.75]);
    });
    record("vector-feather", { what: "a vector mask with a 3 px feather (REFUSED: mask-parameters)" }, function (d) {
      fullLayer(d);
      addVectorMask(d, [rect(20, 4, 44, 60, ADD)]);
      setLayer([["vectorMaskFeather", "#Pxl", 3]]);
    });
    record("group-vector-mask", { what: "a group with a vector mask (REFUSED: group-mask)" }, function (d) {
      var g = d.layerSets.add();
      g.name = "group";
      var a = g.artLayers.add();
      d.activeLayer = a;
      d.selection.selectAll();
      d.selection.fill(rgb(220, 40, 160));
      d.selection.deselect();
      d.activeLayer = g;
      addVectorMask(d, [rect(12, 12, 52, 52, ADD)]);
    });
  } finally {
    app.preferences.maximizeCompatibility = prevCompat;
  }
  return P.finish([]);
})();
