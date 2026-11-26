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

## Person detection

`person_detector` runs YOLO26 on the color stream and publishes
`vision_msgs/Detection2DArray` on `detections` (optionally `detections/image` with drawn boxes).
Parameters live in `config/person_detector.yaml`.

Inference backends (`backend` parameter, `auto` picks the first available):

| Backend | Where | Built when |
|---|---|---|
| `tensorrt` | Jetson, x86 with an NVIDIA GPU | TensorRT + CUDA are found |
| `onnxruntime` | any CPU (devcontainer) | ONNX Runtime is in `/opt/onnxruntime` |

TensorRT builds an engine from the `.onnx` on first start (minutes on the Jetson) and caches it
next to the model as `<model>.<gpu>.trt<version>.fp16.b1.engine`. Engines are GPU- and
version-specific and are not committed.

```bash
# export the model (host venv, see below) -> models/yolo26n.onnx, output [1, 300, 6]
python3 scripts/export_yolo.py models/yolo26n.pt

# live camera
ros2 launch meldog_perception perception.launch.py debug_image:=true
# rosbag instead of the camera
ros2 launch meldog_perception perception.launch.py use_camera:=false

# one image, no ROS: check an export or benchmark a backend
ros2 run meldog_perception detect_image models/yolo26n.onnx image.jpg --backend tensorrt --runs 200
# the same image through Ultralytics, for comparison
python3 scripts/reference_detect.py models/yolo26n.onnx image.jpg
```

## Person tracking

`person_tracker` runs ByteTrack on `detections` and publishes `tracks`
(`vision_msgs/Detection2DArray`, `Detection2D::id` = stable person ID). It logs
`Person N entered` / `Person N left`; with `debug_image:=true` it publishes `tracks/image`
with IDs and velocity arrows. Parameters live in `config/person_tracker.yaml`.

- A person hidden for less than `lost_timeout` (1 s) keeps the ID; after that they get a new one
  (re-identification by appearance is the ReID module's job).
- ByteTrack also uses low-score detections (0.1–0.5) to keep tracks through partial occlusion,
  which is why `person_detector`'s `score_threshold` is 0.1.
- The tracker is plain C++ in `meldog_perception_core` (`tracking/`), with unit tests:
  ```bash
  colcon test --packages-select meldog_perception && colcon test-result --verbose
  ```

## Model export (host, outside the container)

See `scripts/requirements.txt`: export runs in a separate venv with CPU-only torch.
On the Jetson, `.onnx` → `.engine` is done with JetPack's TensorRT.
