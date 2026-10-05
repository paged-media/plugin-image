use image_psd::PsdFile;
#[test]
#[ignore]
fn diffimg() {
    let path = std::env::var("PAGED_FILE").unwrap();
    let out = std::env::var("PAGED_OUT").unwrap();
    let psd = PsdFile::parse(&std::fs::read(&path).unwrap()).unwrap();
    let imp = psd.layer_plates_rgba8().unwrap();
    let stack = image_js::layers::LayerStack::from_psd_plates(&imp).unwrap();
    let ctx = image_conformance::device::test_device().unwrap();
    let ours = pollster::block_on(stack.composite(Some(ctx), None)).unwrap();
    let theirs = psd.composite_rgba8().unwrap().rgba;
    let (w, h) = (imp.width, imp.height);
    let mut both = vec![0u8; (w * 2 * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            for (k, img) in [(0u32, &ours[..]), (1, &theirs[..])] {
                let s = ((y * w + x) * 4) as usize;
                let d = ((y * w * 2 + k * w + x) * 4) as usize;
                both[d..d + 4].copy_from_slice(&img[s..s + 4]);
                both[d + 3] = 255;
            }
        }
    }
    let mut enc = zune_png::PngEncoder::new(
        &both,
        zune_core::options::EncoderOptions::new(
            (w * 2) as usize,
            h as usize,
            zune_core::colorspace::ColorSpace::RGBA,
            zune_core::bit_depth::BitDepth::Eight,
        ),
    );
    let mut buf = Vec::new();
    enc.encode(&mut buf).unwrap();
    std::fs::write(&out, buf).unwrap();
    for (i, l) in stack.layers().iter().enumerate() {
        eprintln!(
            "L{i} {:?} vis={} op={} blend={} group={:?} bounded={:?}",
            l.name,
            l.visible,
            l.opacity,
            l.blend.id,
            l.group,
            l.rgba.bounded().map(|b| b.rect)
        );
    }
    for g in stack.groups() {
        eprintln!(
            "G{} {:?} vis={} pass={} op={} parent={:?}",
            g.id, g.name, g.visible, g.pass_through, g.opacity, g.parent
        );
    }
}
