#!/usr/bin/env python3
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

RECORD = "22921244"
URL = f"https://zenodo.org/records/{RECORD}/files/Supplementary_Materials.zip?download=1"
EXPECTED_MD5 = "edc448cbf8f4cbb4f32c5023d0da9a45"

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
DATA = os.path.join(HERE, "data")


def already_there():
    if not os.path.isdir(ASSETS):
        return False
    pngs = [f for f in os.listdir(ASSETS) if f.lower().endswith(".png")]
    return len(pngs) >= 30 and os.path.exists(
        os.path.join(DATA, "Trading_Study_Results.xlsx"))


def download():
    print(f"downloading record {RECORD} ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "fetch-assets/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        blob = r.read()
    print(f"  {len(blob) / 1e6:.1f} MB")

    got = hashlib.md5(blob).hexdigest()
    if got != EXPECTED_MD5:
        print(f"  ! md5 is {got}, expected {EXPECTED_MD5}", file=sys.stderr)
        print("  ! the record may have been updated; continuing anyway",
              file=sys.stderr)
    return blob


def safe_extract(zf, member, dest_dir, filename):
    name = os.path.basename(filename)
    if not name or name.startswith("."):
        return False
    target = os.path.join(dest_dir, name)
    if os.path.commonpath([os.path.abspath(target), dest_dir]) != dest_dir:
        raise RuntimeError(f"refusing unsafe path: {filename}")
    with zf.open(member) as src, open(target, "wb") as out:
        out.write(src.read())
    return True


def main():
    if already_there():
        print("materials already present, nothing to do")
        return

    blob = download()
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    n_img = n_dat = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            path = member.filename.replace("\\", "/")
            low = path.lower()

            if "/assets/" in low and low.endswith(".png"):
                n_img += safe_extract(zf, member, ASSETS, path)
            elif low.endswith("trading_study_results.xlsx"):
                n_dat += safe_extract(zf, member, DATA, path)

    print(f"extracted {n_img} images -> assets/")
    print(f"extracted {n_dat} data file(s) -> data/")

    if n_img < 30:
        print(f"! expected 30 images, got {n_img}", file=sys.stderr)
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
