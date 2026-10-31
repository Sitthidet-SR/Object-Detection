"""Train a YOLO model on the 5 stationery classes.

Example:
    python train.py                         # default: yolov8n, 100 epochs
    python train.py --model yolov8s.pt --epochs 150 --imgsz 640 --batch 16
    python train.py --device cpu            # no GPU

When training finishes, the best weights are copied to models/best.pt,
which app.py loads automatically.
"""

import argparse
import os
import shutil

import yaml
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_data_yaml(path):
    """Make the dataset 'path' in data.yaml absolute so training works from any cwd."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("path", ".")
    if not os.path.isabs(root):
        cfg["path"] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(path)), root))
    out = os.path.join(BASE_DIR, "runs", "_data_resolved.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(BASE_DIR, "data.yaml"))
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained starting weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'0' for GPU 0, 'cpu' for CPU")
    ap.add_argument("--name", default="stationery")
    args = ap.parse_args()

    data = resolve_data_yaml(args.data)
    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=os.path.join(BASE_DIR, "runs"),
        name=args.name,
        patience=30,
        exist_ok=True,
    )

    metrics = model.val(data=data, project=os.path.join(BASE_DIR, "runs"), name=args.name + "_val", exist_ok=True)
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")

    best = os.path.join(BASE_DIR, "runs", args.name, "weights", "best.pt")
    if os.path.exists(best):
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        dst = os.path.join(BASE_DIR, "models", "best.pt")
        shutil.copy(best, dst)
        print(f"Copied best weights to {dst}")


if __name__ == "__main__":
    main()
