# LRT_meldog_perception

Perception package (ROS 2 Humble) for the Meldog quadruped robot.

## Dev environment (Ubuntu 22.04 + ROS 2 Humble in Docker)

The `.devcontainer/` folder defines the same environment the robot runs (Humble, Ubuntu 22.04),
so you can develop on any host system (e.g. Ubuntu 24.04).

**Host requirements:** Docker (your user in the `docker` group), Cursor or VS Code with the
*Dev Containers* extension, and the librealsense udev rules (`/etc/udev/rules.d/99-realsense-libusb.rules`)
if you want to use the camera.

1. Open the repo in Cursor → `Ctrl+Shift+P` → **Dev Containers: Reopen in Container**.
   The first build takes a few minutes; `rosdep install` runs automatically.
2. Cursor opens the colcon workspace `~/ws`, with this repo at `~/ws/src/meldog_perception`.
3. Build from the container terminal:
   ```bash
   cb   # alias: colcon build --symlink-install (+ compile_commands.json, ccache, Ninja) + source
   ```
   After the first build, run `clangd: Restart language server` once so clangd picks up `compile_commands.json`.
4. Run:
   ```bash
   ros2 launch meldog_perception perception.launch.py
   ```

**GUI (rviz2, rqt):** the container uses the host's X11. Grant access once per host session with
`xhost +SI:localuser:$(whoami)`.

`build/`, `install/`, `log/` and the ccache live in Docker volumes (`meldog-humble-*`), so they
survive container rebuilds. Remove them with `docker volume rm` to start clean.

## Inference backends

OSNet re-identification runs on one of two backends, chosen by the model file extension:

| CMake option | Default | Model | Dependency |
|---|---|---|---|
| `MELDOG_HAS_ONNXRUNTIME` | `ON` | `.onnx` | ONNX Runtime in `/opt/onnxruntime` (or `-DONNXRUNTIME_ROOT=<prefix>`) |
| `MELDOG_HAS_TENSORRT` | `OFF` | `.engine` | CUDA + TensorRT 8.5+ (JetPack on the Jetson) |

The dev container already has ONNX Runtime. To build outside the container, install it the same way
(see the ONNX Runtime step in `.devcontainer/Dockerfile`):

```bash
ORT_VERSION=1.32.0
curl -fsSL -o /tmp/ort.nupkg https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ORT_VERSION}/microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg
python3 -m zipfile -e /tmp/ort.nupkg /tmp/ort_nuget
sudo mkdir -p /opt/onnxruntime-${ORT_VERSION}/include /opt/onnxruntime-${ORT_VERSION}/lib
sudo cp /tmp/ort_nuget/build/native/include/*.h /opt/onnxruntime-${ORT_VERSION}/include/
sudo cp /tmp/ort_nuget/runtimes/linux-x64/native/*.so /opt/onnxruntime-${ORT_VERSION}/lib/   # linux-arm64 on ARM
sudo ln -sf libonnxruntime.so /opt/onnxruntime-${ORT_VERSION}/lib/libonnxruntime.so.1
sudo ln -sfn /opt/onnxruntime-${ORT_VERSION} /opt/onnxruntime
echo /opt/onnxruntime/lib | sudo tee /etc/ld.so.conf.d/onnxruntime.conf && sudo ldconfig
```

`osnet.onnx` keeps its weights in `osnet.onnx.data`: both files must stay in the same directory.

## Model export (host, outside the container)

See `scripts/requirements.txt`: export runs in a separate venv with CPU-only torch.
On the Jetson, `.onnx` → `.engine` is done with JetPack's TensorRT.
