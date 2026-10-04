#!/usr/bin/env bash
# Ask Adobe Photoshop what the right answer to a set of raster operations
# is, and record it. Photoshop is the oracle for paged.image's blend modes,
# adjustments, filters and layer compositing, the way Illustrator is for
# paged.draw's path operations.
#
#   bash scripts/photoshop/run-probe.sh <probe.jsx> [<fixtures-dir>]
#
#   bash scripts/photoshop/run-probe.sh scripts/photoshop/probes/blend-modes.jsx
#
# Maintainer-only (macOS + Photoshop). CI never runs this: it replays the
# committed fixtures through image-conformance/tests/oracle_photoshop.rs
# and psd_composite_photoshop.rs.
#
# In order -- each step exists because of a trap (see README.md):
#   1. generates the stimuli (lib/stimuli.mjs) into a STAGING dir under
#      $TMPDIR, which Photoshop can read and write without a prompt;
#   2. asks macOS (without prompting) whether this shell may control
#      Photoshop, and says so in words when it may not;
#   3. brings Photoshop up and PINGS it under a short timeout -- an
#      unanswered consent prompt, a sign-in screen and a modal dialog all
#      look like a silent hang;
#   4. hands Photoshop the probe as TEXT (prelude + lib/probe-lib.jsx +
#      the probe) and captures the JSON it returns;
#   5. pings again and compares open-document counts -- a probe that left
#      a dialog or a document open is not a recording;
#   6. judges the ARTIFACT (lib/write-fixture.mjs: every case answered,
#      every output a 16-bit PNG of the stimulus size) and only then
#      copies stimuli + outputs into the fixtures dir with provenance.
#
# Environment:
#   PHOTOSHOP_BUNDLE_ID        default com.adobe.Photoshop
#   PAGED_PROBE_PING_TIMEOUT   seconds to wait for the ping   (default 60)
#   PAGED_PROBE_TIMEOUT        seconds to wait for the probe  (default 900)
#   PAGED_PROBE_STAGE          staging dir (default: mktemp under $TMPDIR,
#                              removed on success, KEPT on failure)
#
# Exit codes: 0 recorded / 2 usage / 3 Photoshop did not answer /
#             4 Photoshop answered with something that is not a recording.
set -euo pipefail

die() {
  local code="$1"
  shift
  printf '\nrun-probe: FAILED -- %s\n' "$1" >&2
  shift
  for line in "$@"; do printf '  %s\n' "$line" >&2; done
  exit "$code"
}

[ $# -ge 1 ] && [ $# -le 2 ] || die 2 "usage: run-probe.sh <probe.jsx> [<fixtures-dir>]"
[ "$(uname -s)" = "Darwin" ] || die 2 "this drives Adobe Photoshop through osascript; macOS only"
[ -f "$1" ] || die 2 "no such probe: $1"
command -v osascript >/dev/null || die 2 "osascript not found"
command -v node >/dev/null || die 2 "node not found"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROBE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
NAME="$(basename "$PROBE" .jsx)"
OUTDIR="${2:-$ROOT/image-conformance/fixtures/photoshop}"
LIB="$ROOT/scripts/photoshop/lib/probe-lib.jsx"
WRITER="$ROOT/scripts/photoshop/lib/write-fixture.mjs"
STIM="$ROOT/scripts/photoshop/lib/stimuli.mjs"
BUNDLE_ID="${PHOTOSHOP_BUNDLE_ID:-com.adobe.Photoshop}"
PING_TIMEOUT="${PAGED_PROBE_PING_TIMEOUT:-60}"
RUN_TIMEOUT="${PAGED_PROBE_TIMEOUT:-900}"

for f in "$LIB" "$PROBE"; do
  n="$(LC_ALL=C tr -d '\11\12\15\40-\176' <"$f" | wc -c | tr -d ' ')"
  [ "$n" -eq 0 ] || die 2 "$n non-ASCII byte(s) in $f" "ExtendScript sources must be plain ASCII."
done

if [ -n "${PAGED_PROBE_STAGE:-}" ]; then
  STAGE="$PAGED_PROBE_STAGE"
  OWN_STAGE=0
else
  STAGE="$(mktemp -d "${TMPDIR:-/tmp}/paged-photoshop-probe.XXXXXX")"
  OWN_STAGE=1
fi
mkdir -p "$STAGE/$NAME"
SRC="$STAGE/$NAME.combined.jsx"
RAW="$STAGE/$NAME.raw.json"
ERR="$STAGE/$NAME.osascript.err"

# --- 1. stimuli + the script Photoshop will be handed -----------------------
node "$STIM" "$STAGE/stimuli" >/dev/null
{
  printf 'var PAGED_STAGE = "%s";\n' "$STAGE"
  cat "$LIB"
  printf '\n'
  cat "$PROBE"
} >"$SRC"

# --- 2. may this shell control Photoshop? ------------------------------------
#      0 allowed / -1744 not decided (macOS will prompt) / -1743 denied /
#   -600 not running
consent_status() {
  osascript -l JavaScript - "$BUNDLE_ID" 2>/dev/null <<'JXA' || echo unknown
ObjC.import('Foundation');
ObjC.import('CoreServices');
ObjC.bindFunction('AEDeterminePermissionToAutomateTarget',
  ['int', ['void *', 'unsigned int', 'unsigned int', 'bool']]);
function run(argv) {
  var d = $.NSAppleEventDescriptor.descriptorWithBundleIdentifier(argv[0]);
  return String($.AEDeterminePermissionToAutomateTarget(d.aeDesc, 0x2a2a2a2a, 0x2a2a2a2a, false));
}
JXA
}

CONSENT_HELP=(
  "macOS decides per controlling app whether it may send Apple events to"
  "Photoshop. Grant it: answer the system prompt naming Photoshop with"
  "\"Allow\", or System Settings > Privacy & Security > Automation > <the"
  "app that owns this shell> > Adobe Photoshop. Then re-run."
)

echo "run-probe: $NAME (stage $STAGE)"
if ! pgrep -qf "Adobe Photoshop" ; then
  open -b "$BUNDLE_ID" 2>"$ERR" || die 3 "could not launch $BUNDLE_ID" "$(cat "$ERR")"
  for _ in $(seq 1 120); do
    [ "$(consent_status)" != "-600" ] && break
    sleep 2
  done
fi
case "$(consent_status)" in
  0 | unknown) ;;
  -1743) die 3 "macOS has DENIED this shell permission to control Photoshop (-1743)" "${CONSENT_HELP[@]}" ;;
  -1744) echo "run-probe: macOS will now ask whether this shell may control Photoshop -- answer within ${PING_TIMEOUT}s" ;;
  -600) die 3 "Photoshop ($BUNDLE_ID) did not come up within 240 s" ;;
esac

ping_app() {
  osascript 2>"$ERR" <<OSA
with timeout of $PING_TIMEOUT seconds
    tell application id "$BUNDLE_ID"
        do javascript "$1"
    end tell
end timeout
OSA
}

explain_no_answer() {
  local err
  err="$(cat "$ERR" 2>/dev/null || true)"
  case "$err" in
    *-1743*) die 3 "macOS refused the Apple event to Photoshop (-1743)" "$err" "${CONSENT_HELP[@]}" ;;
    *-1712*) die 3 "Photoshop did not answer within ${PING_TIMEOUT}s (-1712)" "$err" \
      "It is most likely showing a MODAL dialog (sign-in, licence, crash" \
      "recovery, a script alert) or an unanswered consent prompt. Dismiss it" \
      "in the app and re-run. Do not SIGKILL it." ;;
    *) die 3 "Photoshop did not answer \"$1\"" "$err" ;;
  esac
}

VERSION="$(ping_app "app.version")" || explain_no_answer "app.version"
[ -n "$VERSION" ] || die 3 "Photoshop answered the version ping with nothing"
DOCS_BEFORE="$(ping_app "String(app.documents.length)")" || explain_no_answer "app.documents.length"
echo "run-probe: Photoshop $VERSION answers; $DOCS_BEFORE document(s) open"

# --- 4. run the probe --------------------------------------------------------
rc=0
osascript - "$SRC" >"$RAW" 2>"$ERR" <<OSA || rc=$?
on run argv
    set src to read (POSIX file (item 1 of argv)) as «class utf8»
    with timeout of $RUN_TIMEOUT seconds
        tell application id "$BUNDLE_ID"
            return do javascript src
        end tell
    end timeout
end run
OSA
[ "$rc" -eq 0 ] || die 3 "the probe did not return (osascript exit $rc)" "$(cat "$ERR")" \
  "Staging kept: $STAGE"

# --- 5. still answering, nothing left open? ---------------------------------
DOCS_AFTER="$(ping_app "String(app.documents.length)")" ||
  die 3 "Photoshop stopped answering AFTER the probe (a modal dialog?)" "$(cat "$ERR")" "Staging kept: $STAGE"
[ "$DOCS_AFTER" = "$DOCS_BEFORE" ] ||
  die 4 "the probe left documents open: $DOCS_BEFORE before, $DOCS_AFTER after" \
    "Close them in Photoshop WITHOUT saving. Staging kept: $STAGE"

# --- 6. judge the artifact, then write the fixture ---------------------------
SHA="$(shasum -a 256 <"$SRC" | cut -d' ' -f1)"
node "$WRITER" "$RAW" "$STAGE" "$OUTDIR" "$NAME" "${PROBE#"$ROOT"/}" "$SHA" "$(sw_vers -productVersion)" ||
  die 4 "Photoshop's reply was refused; nothing was written" "Staging kept: $STAGE"

[ "$OWN_STAGE" -eq 1 ] && rm -rf "$STAGE"
echo "==> $OUTDIR/$NAME.photoshop.json"
