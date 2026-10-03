#!/usr/bin/env bash
# Builds handin/Andreas-Olsson.zip, the Canvas hand-in: the slides and
# everything needed to reproduce them (package, tests, notebooks,
# results, figures, readme, the uv project files), under one folder.
# Only files git tracks go in, so data/, .venv/ and caches never do,
# and the zip is the same from any clean checkout. Run from anywhere:
#   bash handin/build.sh
set -euo pipefail

lab="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$lab"
out="handin/Andreas-Olsson.zip"

test -f handin/slides.pptx || { echo "handin/slides.pptx is missing" >&2; exit 1; }

# instructions/ is the course's own material, not part of the answer
git ls-files -- . ':!:instructions' ':!:data' ':!:handin/*.zip' > /tmp/handin-files.$$
trap 'rm -f /tmp/handin-files.$$' EXIT

rm -f "$out"
python3 - "$out" /tmp/handin-files.$$ <<'PY'
import sys, zipfile
out, listing = sys.argv[1], sys.argv[2]
files = [line.strip() for line in open(listing) if line.strip()]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f"Andreas-Olsson/{f}")
print(f"{len(files)} files in {out}")
PY
