// MASK PARAMETERS, written by Photoshop itself: density and feather on a
// user mask and on a vector mask. Photoshop applies them live, outside
// every stored mask channel, so only the merged composite shows them; the
// replay (psd_mask_parameters.rs) holds our application of them to it.
(function () {
  var P = PagedProbe;
  var st = P.begin("mask-parameters");
  var S = 64;
  var DIR = PAGED_STAGE + "/mask-parameters/";
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
      c.files = ["mask-parameters/" + id + ".psd", "mask-parameters/" + id + ".png"];
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
    record("vector-feather-3", { vector_feather: 3 }, function (d) {
      fullLayer(d);
      addVectorMask(d, [rect(20, 4, 44, 60, ADD)]);
      setLayer([["vectorMaskFeather", "#Pxl", 3]]);
    });
    record("vector-feather-7", { vector_feather: 7.5 }, function (d) {
      fullLayer(d);
      addVectorMask(d, [rect(14, 10, 50, 54, ADD)]);
      setLayer([["vectorMaskFeather", "#Pxl", 7.5]]);
    });
    record("vector-density-60", { vector_density: 60 }, function (d) {
      fullLayer(d);
      addVectorMask(d, [rect(20, 4, 44, 60, ADD)]);
      setLayer([["vectorMaskDensity", "#Prc", 60]]);
    });
    record("user-density-60", { user_density: 60 }, function (d) {
      fullLayer(d);
      userMask(d, 16, 16, 48, 48, 0);
      setLayer([["userMaskDensity", "#Prc", 60]]);
    });
    record("user-feather-4", { user_feather: 4 }, function (d) {
      fullLayer(d);
      userMask(d, 16, 16, 48, 48, 0);
      setLayer([["userMaskFeather", "#Pxl", 4]]);
    });
    record("user-density-and-feather", { user_density: 80, user_feather: 2.5 }, function (d) {
      fullLayer(d);
      userMask(d, 12, 20, 40, 56, 0);
      setLayer([["userMaskDensity", "#Prc", 80], ["userMaskFeather", "#Pxl", 2.5]]);
    });
  } finally {
    app.preferences.maximizeCompatibility = prevCompat;
  }
  return P.finish([]);
})();
