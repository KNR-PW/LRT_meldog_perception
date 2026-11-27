#!/usr/bin/env python3
"""
Export YOLO weights for the perception node into models/.

    --format onnx    .pt  -> <name>.onnx                  (CPU / ONNX Runtime)
    --format engine  .pt  -> <name>.onnx -> <name>.engine (TensorRT, FP16)
                     .onnx -> <name>.engine               (no torch needed)

A TensorRT engine only loads on the GPU and TensorRT version it was built
with, so build it on the target device (Jetson) from the .onnx.

Example::

    # dev machine
    python3 scripts/export_yolo.py models/yolo26n.pt
    # Jetson
    python3 scripts/export_yolo.py models/yolo26n.onnx --format engine
"""

import argparse
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export YOLO .pt to .onnx / .engine')
    parser.add_argument('weights', type=Path, help='.pt weights or an exported .onnx')
    parser.add_argument('--format', choices=['onnx', 'engine'], default='onnx')
    parser.add_argument('--name', default=None, help='output base name (default: weights stem)')
    parser.add_argument('--out', type=Path, default=MODELS_DIR, help='output directory')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[480, 640], metavar=('H', 'W'),
                        help='network input height and width, multiples of 32')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset')
    parser.add_argument('--fp32', action='store_true', help='build the engine in FP32 instead of FP16')
    return parser.parse_args()


def export_onnx(pt_path: Path, onnx_path: Path, imgsz: list[int], opset: int) -> None:
    from ultralytics import YOLO

    # nms=False keeps YOLO26's end-to-end head: output [1, 300, 6] rows of
    # x1 y1 x2 y2 score class, no NMS needed in C++. Without it the export
    # falls back to the one-to-many head [1, 84, anchors] that needs NMS.
    exported = YOLO(str(pt_path)).export(
        format='onnx', imgsz=imgsz, opset=opset, simplify=True, device='cpu', nms=False
    )
    shutil.move(exported, onnx_path)


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'Failed to parse {onnx_path}:\n{errors}')

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT engine build failed')
    engine_path.write_bytes(engine)


def main() -> None:
    args = parse_args()
    name = args.name or args.weights.stem
    args.out.mkdir(parents=True, exist_ok=True)

    if args.weights.suffix == '.onnx':
        onnx_path = args.weights
    else:
        onnx_path = args.out / f'{name}.onnx'
        export_onnx(args.weights, onnx_path, args.imgsz, args.opset)
        print(f'ONNX:   {onnx_path}')

    if args.format == 'engine':
        engine_path = args.out / f'{name}.engine'
        build_engine(onnx_path, engine_path, fp16=not args.fp32)
        print(f'Engine: {engine_path}')


if __name__ == '__main__':
    main()
