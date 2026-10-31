"""Collect training images from the webcam.

    python tools/collect_images.py --out dataset/raw

Keys:  1-5 = save a frame for Pen/Pencil/Eraser/Ruler/Scissors
       0   = save a frame with several objects (mixed)
       q   = quit
Images are saved to <out>/<ClassName>/xxx.jpg. Label them afterwards
(Roboflow / CVAT / LabelImg) in YOLO format.
"""

import argparse
import os
import time

import cv2

CLASSES = ["Pen", "Pencil", "Eraser", "Ruler", "Scissors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/raw")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    counts = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        cv2.putText(view, "1 Pen 2 Pencil 3 Eraser 4 Ruler 5 Scissors 0 Mixed q Quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y = 50
        for k, v in counts.items():
            cv2.putText(view, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 22
        cv2.imshow("collect", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("0") <= key <= ord("5"):
            name = "Mixed" if key == ord("0") else CLASSES[key - ord("1")]
            folder = os.path.join(args.out, name)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{name}_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(path, frame)
            counts[name] = counts.get(name, 0) + 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
