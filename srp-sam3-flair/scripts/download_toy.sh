#!/bin/bash
# Fetch the FLAIR-HUB toy subset (~750 MB) -- enough to build and verify the loader
# without committing to the full download.
#
# Scale reference before you reach for the full thing:
#   FLAIR-HUB (all modalities, incl. Sentinel-1/2 and historical panchromatic): ~750 GB
#   FLAIR #1 (aerial patches + masks only): far smaller, and all this project needs
#
# Licence: Open Licence 2.0 / etalab-2.0. Free to use and redistribute with attribution
# to IGN. Compatible with CC-BY. Cite the FLAIR papers in any write-up.
set -euo pipefail

DEST="${1:-./data}"
URL="https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_hub/FLAIR-HUB_TOY_DATASET.zip"
ARCHIVE="$DEST/FLAIR-HUB_TOY_DATASET.zip"

mkdir -p "$DEST"

if [ -f "$ARCHIVE" ]; then
    echo "Archive already present at $ARCHIVE -- skipping download."
else
    echo "Downloading FLAIR-HUB toy dataset (~750 MB) to $DEST ..."
    curl -L --fail --progress-bar -o "$ARCHIVE" "$URL"
fi

echo "Extracting ..."
unzip -q -o "$ARCHIVE" -d "$DEST"

echo
echo "Done. Now verify the loader's assumptions against the real files:"
echo "    python scripts/inspect_flair.py --root $DEST"
