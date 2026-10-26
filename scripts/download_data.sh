#!/usr/bin/env bash
# Download the four datasets of the thesis into data/ and bring them into the layout used by
# launch/protocol.sh.
#
#   bash scripts/download_data.sh all            # everything (about 0.7 GB of downloads)
#   bash scripts/download_data.sh openea         # EN-DE-15K and EN-DE-100K (OpenEA v2.0)
#   bash scripts/download_data.sh dbp1m          # DBP1M EN-DE (LargeEA release)
#   bash scripts/download_data.sh fb_dbp2m       # FB-DBP2M (distributed with the DivEA data)
#
# Needs curl and, for DBP1M (hosted on Google Drive), gdown (`uvx gdown` or `pip install gdown`).
# DBP1M and FB-DBP2M are converted to the OpenEA layout and split 30/10/60 with seed 42, which
# reproduces the splits of the thesis exactly. EN-DE-100K uses OpenEA's own 721_5fold/1.
# EN-DE-15K gets a 30/10/60 custom_30pct_fold/1 split from scripts/make_30pct_fold.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
mkdir -p data

OPENEA_URL="https://ndownloader.figshare.com/files/34234391"   # figshare, OpenEA v2.0
OPENEA_MD5="92fc8bd7f28959625a316f687b0affb2"
DBP1M_GDRIVE_ID="15jeGD-6pVGlqI5jCn7KJfGIER6AeoQ-L"            # LargeEA's DBP1M release
DIVEA_URL="https://www.dropbox.com/scl/fo/j95qpgb4cpxjcu1am3koy/ADnyA1olK-S5oGrbfV1Cb18?rlkey=cs89tuhnwl2hmvcy2sk5r86nm&dl=1"

md5_of() { "$PYTHON" -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

get_openea() {
    if [ ! -d data/OpenEA_dataset_v2.0 ]; then
        echo ">> OpenEA v2.0 (~237 MB)"
        curl -L --retry 3 --progress-bar "$OPENEA_URL" -o data/OpenEA_dataset_v2.0.zip
        [ "$(md5_of data/OpenEA_dataset_v2.0.zip)" = "$OPENEA_MD5" ] || { echo "MD5 mismatch" >&2; exit 1; }
        "$PYTHON" scripts/extract_zip.py data/OpenEA_dataset_v2.0.zip data
        rm -f data/OpenEA_dataset_v2.0.zip
    fi
    local d=data/OpenEA_dataset_v2.0/EN_DE_15K_V1
    [ -d "$d/custom_30pct_fold/1" ] || "$PYTHON" scripts/make_30pct_fold.py --data_dir "$d"
}

get_dbp1m() {
    if [ ! -d data/mkdata ]; then
        echo ">> DBP1M (Google Drive)"
        if command -v gdown >/dev/null; then GDOWN=gdown; else GDOWN="uvx gdown"; fi
        $GDOWN "$DBP1M_GDRIVE_ID" -O data/DBP1M.zip
        "$PYTHON" scripts/extract_zip.py data/DBP1M.zip data
        rm -f data/DBP1M.zip
    fi
    [ -d data/DBP1M_DE ] || "$PYTHON" scripts/prepare_dbp1m.py --mkdata_dir data/mkdata --lang de --out_dir data/DBP1M_DE
    [ -d data/DBP1M_DE/316_1fold ] || "$PYTHON" scripts/split_openea.py --data_dir data/DBP1M_DE --train 0.3 --valid 0.1 --folds 1
}

get_fb_dbp2m() {
    if [ ! -d data/datasets/2m/fb_dbp ]; then
        echo ">> FB-DBP2M (DivEA data, ~187 MB)"
        curl -L --retry 3 --progress-bar "$DIVEA_URL" -o data/DivEA_data.zip
        "$PYTHON" scripts/extract_zip.py data/DivEA_data.zip data --nested
        rm -f data/DivEA_data.zip
    fi
    [ -d data/FB_DBP2M ] || "$PYTHON" scripts/prepare_fb_dbp2m.py --src_dir data/datasets/2m/fb_dbp --out_dir data/FB_DBP2M
    [ -d data/FB_DBP2M/316_1fold ] || "$PYTHON" scripts/split_openea.py --data_dir data/FB_DBP2M --train 0.3 --valid 0.1 --folds 1
}

case "${1:-all}" in
    openea)   get_openea ;;
    dbp1m)    get_dbp1m ;;
    fb_dbp2m) get_fb_dbp2m ;;
    all)      get_openea; get_dbp1m; get_fb_dbp2m ;;
    *) echo "usage: $0 [all|openea|dbp1m|fb_dbp2m]" >&2; exit 1 ;;
esac
echo "done."
