#!/usr/bin/env bash
# Builds a TensorRT engine from an .onnx model with trtexec and prints its speed.
# Run it on the device that will use the engine (Jetson): engines are specific to
# the GPU and the TensorRT version.
#
#   scripts/build_engine.sh models/yolo26n.onnx              # FP16 -> models/yolo26n.engine
#   scripts/build_engine.sh models/yolo26n.onnx --fp32       # FP32 -> models/yolo26n.fp32.engine
#   scripts/build_engine.sh models/yolo26n.engine            # only benchmark an existing engine
#
# Extra arguments go to trtexec, e.g. a batch range for a dynamic-batch OSNet:
#   scripts/build_engine.sh models/osnet.onnx --minShapes=input:1x3x256x128 \
#       --optShapes=input:4x3x256x128 --maxShapes=input:8x3x256x128
set -euo pipefail

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if ! command -v "$TRTEXEC" >/dev/null 2>&1; then
    TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" ]]; then
    echo "trtexec not found (JetPack: /usr/src/tensorrt/bin/trtexec). Set TRTEXEC=<path>." >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

model="$1"
shift

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

case "$model" in
    *.onnx)
        precision="--fp16"
        engine="${model%.onnx}.engine"
        if [[ "${1:-}" == "--fp32" ]]; then
            precision=""
            engine="${model%.onnx}.fp32.engine"
            shift
        fi
        echo "Building $engine from $model (${precision:-fp32}), this takes a few minutes on the Jetson..."
        # shellcheck disable=SC2086
        "$TRTEXEC" --onnx="$model" --saveEngine="$engine" $precision "$@" >"$log" 2>&1 \
            || { tail -n 30 "$log" >&2; exit 1; }
        echo "Saved $engine"
        ;;
    *.engine)
        "$TRTEXEC" --loadEngine="$model" "$@" >"$log" 2>&1 || { tail -n 30 "$log" >&2; exit 1; }
        ;;
    *)
        echo "Expected an .onnx or .engine file, got $model" >&2
        exit 1
        ;;
esac

# Only the speed summary of trtexec's long output.
grep -E "Throughput:|Latency:|GPU Compute Time:" "$log" | sed 's/^\[[^]]*\] \[I\] //'
