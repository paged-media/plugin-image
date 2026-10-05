use image_psd::PsdFile;
#[test]
#[ignore]
fn layerdump() {
    let path = std::env::var("PAGED_FILE").unwrap();
    let want = std::env::var("PAGED_LAYER").unwrap();
    let psd = PsdFile::parse(&std::fs::read(&path).unwrap()).unwrap();
    for l in &psd.layer_mask.layers {
        if !l.name().contains(&want) {
            continue;
        }
        let keys: Vec<String> = l
            .addl
            .iter()
            .map(|a| {
                format!(
                    "{}({})",
                    String::from_utf8_lossy(&a.key),
                    a.raw_block.as_ref().map_or(0, |b| b.len())
                )
            })
            .collect();
        eprintln!(
            "{:?} blend={} op={} flags={:#x} clip={} ch={:?} mask={:?} ranges={:?} keys={:?}",
            l.name(),
            String::from_utf8_lossy(&l.blend_key),
            l.opacity,
            l.flags,
            l.clipping,
            l.channels.iter().map(|c| c.id).collect::<Vec<_>>(),
            l.mask.as_ref().map(|m| (m.flags, m.parameters())),
            &l.blend_ranges.raw.iter().take(16).collect::<Vec<_>>(),
            keys
        );
        for a in &l.addl {
            if &a.key == b"lfx2" {
                if let Ok(fx) = image_psd::effects::Effects::parse_lfx2(
                    a.raw_block
                        .as_deref()
                        .and_then(|b| b.get(12..))
                        .unwrap_or(&[]),
                ) {
                    eprintln!("   fx {:?}", fx);
                }
            }
        }
    }
}
