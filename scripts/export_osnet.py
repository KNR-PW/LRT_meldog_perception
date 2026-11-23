#!/usr/bin/env python3
"""
Export OSNet weights for the perception node into models/.

    --format onnx    .pth  -> <name>.onnx                  (CPU / ONNX Runtime)
    --format engine  .pth  -> <name>.onnx -> <name>.engine (TensorRT, FP16)
                     .onnx -> <name>.engine                (no torch needed)

A TensorRT engine only loads on the GPU and TensorRT version it was built
with, so build it on the target device (Jetson) from the .onnx.

Example::

    # dev machine
    python3 scripts/export_osnet.py models/osnet_ain_x1_0_msmt17.pth --arch osnet_ain_x1_0
    # Jetson
    python3 scripts/export_osnet.py models/osnet_ain_x1_0_msmt17.onnx --format engine
"""

import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export OSNet .pth to .onnx / .engine')
    parser.add_argument('weights', type=Path, help='.pth weights or an exported .onnx')
    parser.add_argument('--arch', default='osnet_ain_x1_0', help='torchreid model name')
    parser.add_argument('--format', choices=['onnx', 'engine'], default='onnx')
    parser.add_argument('--name', default=None, help='output base name (default: weights stem)')
    parser.add_argument('--out', type=Path, default=MODELS_DIR, help='output directory')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[256, 128], metavar=('H', 'W'),
                        help='network input height and width')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset')
    parser.add_argument('--fp32', action='store_true', help='build the engine in FP32 instead of FP16')
    return parser.parse_args()


def export_onnx(pth_path: Path, onnx_path: Path, arch: str, imgsz: list[int], opset: int) -> None:
    import torch
    import torchreid

    model = torchreid.models.build_model(name=arch, num_classes=1000, pretrained=False)
    torchreid.utils.load_pretrained_weights(model, str(pth_path))
    model.eval()

    torch.onnx.export(
        model, torch.rand(1, 3, *imgsz), str(onnx_path), opset_version=opset,
        input_names=['input'], output_names=['features'],
        dynamic_axes={'input': {0: 'batch'}, 'features': {0: 'batch'}},
    )


def build_engine(onnx_path: Path, engine_path: Path, imgsz: list[int], fp16: bool) -> None:
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

    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 3, *imgsz), (4, 3, *imgsz), (8, 3, *imgsz))
    config.add_optimization_profile(profile)

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
        export_onnx(args.weights, onnx_path, args.arch, args.imgsz, args.opset)
        print(f'ONNX:   {onnx_path}')

    if args.format == 'engine':
        engine_path = args.out / f'{name}.engine'
        build_engine(onnx_path, engine_path, args.imgsz, fp16=not args.fp32)
        print(f'Engine: {engine_path}')


if __name__ == '__main__':
    main()
