"""Split labeled images into train/val folders for YOLO.

    python tools/split_dataset.py --src labeled --dst dataset --val 0.2

<src> must contain images (*.jpg/*.png) with a YOLO .txt label of the same
name next to each image (or in <src>/labels). Output:
    dataset/images/{train,val}, dataset/labels/{train,val}
"""

import argparse
import glob
import os
import random
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def find_label(img_path, src):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    for cand in (os.path.splitext(img_path)[0] + ".txt", os.path.join(src, "labels", stem + ".txt")):
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="dataset")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = [p for p in glob.glob(os.path.join(args.src, "**", "*"), recursive=True) if p.lower().endswith(IMG_EXT)]
    pairs = [(p, find_label(p, args.src)) for p in images]
    pairs = [(i, l) for i, l in pairs if l]
    if not pairs:
        raise SystemExit("No image/label pairs found")

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.dst, sub, split), exist_ok=True)
        for img, lbl in items:
            shutil.copy(img, os.path.join(args.dst, "images", split, os.path.basename(img)))
            stem = os.path.splitext(os.path.basename(img))[0]
            shutil.copy(lbl, os.path.join(args.dst, "labels", split, stem + ".txt"))
        print(f"{split}: {len(items)} images")


if __name__ == "__main__":
    main()
