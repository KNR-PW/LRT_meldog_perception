#!/usr/bin/env python3
"""
Reference detections from Ultralytics for checking the C++ detector.

Runs the same .onnx model on an image and prints boxes in the format of
`ros2 run meldog_perception detect_image`, so the two outputs can be compared
line by line (expect differences of about a pixel from resizing).

    python3 scripts/reference_detect.py models/yolo26n.onnx bus.jpg
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Ultralytics reference detections')
    parser.add_argument('model', type=Path, help='.onnx (or .pt) model')
    parser.add_argument('image', type=Path)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--all-classes', action='store_true', help='keep every class, not only person')
    args = parser.parse_args()

    import onnx
    from ultralytics import YOLO

    imgsz = None
    if args.model.suffix == '.onnx':
        meta = {p.key: p.value for p in onnx.load(str(args.model)).metadata_props}
        imgsz = eval(meta['imgsz'])  # noqa: S307 - '[480, 640]' written by the Ultralytics exporter

    result = YOLO(str(args.model), task='detect').predict(
        str(args.image), imgsz=imgsz, conf=args.threshold, classes=None if args.all_classes else [0],
        device='cpu', verbose=False,
    )[0]

    boxes = sorted(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()),
                   key=lambda item: -item[1])
    print(f'{len(boxes)} detections')
    for (x1, y1, x2, y2), score, cls in boxes:
        print(f'  class {int(cls):2d}  score {score:.3f}  box [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]')


if __name__ == '__main__':
    main()
