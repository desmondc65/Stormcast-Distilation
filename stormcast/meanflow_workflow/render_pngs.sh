#!/usr/bin/env bash
# Render the architecture SVGs to 2x PNGs with headless Chrome.
#
# Chrome headless has a quirk where the last SVG element may not paint when
# the window height exactly matches the SVG height, so we render with extra
# height and crop back to the exact canvas afterwards.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../stormcast_env/bin/python

for svg in regression_model.svg meanflow_model.svg; do
    w=$(grep -o 'width="[0-9]*"' "$svg" | head -1 | grep -o '[0-9]*')
    h=$(grep -o 'height="[0-9]*"' "$svg" | head -1 | grep -o '[0-9]*')
    png="${svg%.svg}.png"
    google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
        --default-background-color=FFFFFFFF --screenshot="$png" \
        --window-size="$w,$((h + 200))" --force-device-scale-factor=2 \
        "file://$PWD/$svg" >/dev/null 2>&1
    "$PY" - "$png" "$w" "$h" <<'EOF'
import sys
from PIL import Image
p, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
im = Image.open(p)
im.crop((0, 0, w * 2, h * 2)).save(p)
print(f"{p}: {w * 2}x{h * 2}")
EOF
done
