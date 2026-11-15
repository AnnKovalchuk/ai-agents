#!/usr/bin/env python3
import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("needs Pillow:  pip install pillow")

HERE = os.path.dirname(os.path.abspath(__file__))


def mask_one(path, out_path, h_frac, w_frac):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    box_w, box_h = int(w * w_frac), int(h * h_frac)

    bg = im.getpixel((min(box_w + 5, w - 1), max(box_h // 2, 1)))
    im.paste(bg, (0, 0, box_w, box_h))
    im.save(out_path)
    return box_w, box_h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(HERE, "assets"))
    p.add_argument("--dst", default=os.path.join(HERE, "assets_masked"))
    p.add_argument("--height", type=float, default=0.10,
                   help="fraction of image height to cover (default 0.10)")
    p.add_argument("--width", type=float, default=0.55,
                   help="fraction of image width to cover (default 0.55); the "
                        "price axis is on the right and must stay visible")
    a = p.parse_args()

    if not os.path.isdir(a.src):
        sys.exit(f"{a.src} not found -- run fetch_assets.py first")
    os.makedirs(a.dst, exist_ok=True)

    n = 0
    for name in sorted(os.listdir(a.src)):
        if "_start" not in name.lower() or not name.lower().endswith((".png", ".jpg")):
            continue
        bw, bh = mask_one(os.path.join(a.src, name),
                          os.path.join(a.dst, name), a.height, a.width)
        n += 1

    print(f"masked {n} charts -> {a.dst}  (patch {bw}x{bh} px)")
    if n != 15:
        print(f"! expected 15 start charts, found {n}", file=sys.stderr)
    print("\nCheck one by eye before running: the instrument name must be gone "
          "and the candles untouched.")


if __name__ == "__main__":
    main()
