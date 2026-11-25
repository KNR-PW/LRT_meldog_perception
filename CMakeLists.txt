cmake_minimum_required(VERSION 3.16)
project(meldog_perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

# ---------------------------------------------------------------------------
# Inference backends. Each one is optional; at least one must be found.
#   TensorRT:     Jetson (JetPack) or an x86 machine with an NVIDIA GPU
#   ONNX Runtime: CPU fallback for development (installed in the devcontainer)
# ---------------------------------------------------------------------------
option(MELDOG_WITH_TENSORRT "Build the TensorRT backend when TensorRT is found" ON)
option(MELDOG_WITH_ONNXRUNTIME "Build the ONNX Runtime backend when it is found" ON)
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime install prefix")

set(MELDOG_HAS_TENSORRT OFF)
if(MELDOG_WITH_TENSORRT)
  find_package(CUDAToolkit QUIET)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include /usr/include/${CMAKE_LIBRARY_ARCHITECTURE} /usr/local/tensorrt/include)
  find_library(TENSORRT_NVINFER nvinfer
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE} /usr/local/tensorrt/lib)
  find_library(TENSORRT_NVONNXPARSER nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE} /usr/local/tensorrt/lib)
  if(CUDAToolkit_FOUND AND TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER AND TENSORRT_NVONNXPARSER)
    set(MELDOG_HAS_TENSORRT ON)
  endif()
endif()

set(MELDOG_HAS_ONNXRUNTIME OFF)
if(MELDOG_WITH_ONNXRUNTIME)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(MELDOG_HAS_ONNXRUNTIME ON)
  endif()
endif()

message(STATUS "meldog_perception backends: TensorRT=${MELDOG_HAS_TENSORRT} ONNXRuntime=${MELDOG_HAS_ONNXRUNTIME}")
if(NOT MELDOG_HAS_TENSORRT AND NOT MELDOG_HAS_ONNXRUNTIME)
  message(FATAL_ERROR "No inference backend found. Install TensorRT (JetPack) or ONNX Runtime "
                      "(set ONNXRUNTIME_ROOT, default /opt/onnxruntime).")
endif()

# --- core library: inference backends + detector (no ROS) -------------------
add_library(meldog_perception_core SHARED
  src/inference/inference_backend.cpp
  src/yolo_detector.cpp
)
target_include_directories(meldog_perception_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(meldog_perception_core PUBLIC ${OpenCV_LIBS})

if(MELDOG_HAS_TENSORRT)
  target_sources(meldog_perception_core PRIVATE src/inference/tensorrt_backend.cpp)
  target_compile_definitions(meldog_perception_core PRIVATE MELDOG_HAS_TENSORRT)
  target_include_directories(meldog_perception_core PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(meldog_perception_core PRIVATE ${TENSORRT_NVINFER} ${TENSORRT_NVONNXPARSER} CUDA::cudart)
endif()

if(MELDOG_HAS_ONNXRUNTIME)
  target_sources(meldog_perception_core PRIVATE src/inference/onnxruntime_backend.cpp)
  target_compile_definitions(meldog_perception_core PRIVATE MELDOG_HAS_ONNXRUNTIME)
  target_include_directories(meldog_perception_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(meldog_perception_core PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

# --- ROS nodes ----------------------------------------------------------------
add_library(person_detector_component SHARED src/person_detector_node.cpp)
target_link_libraries(person_detector_component meldog_perception_core)
ament_target_dependencies(person_detector_component
  rclcpp rclcpp_components sensor_msgs vision_msgs cv_bridge)
rclcpp_components_register_node(person_detector_component
  PLUGIN "meldog_perception::PersonDetectorNode"
  EXECUTABLE person_detector)

add_executable(camera_listener src/camera_listener.cpp)
target_include_directories(camera_listener PRIVATE include)
ament_target_dependencies(camera_listener rclcpp sensor_msgs)

# --- tools ----------------------------------------------------------------------
add_executable(detect_image src/tools/detect_image.cpp)
target_link_libraries(detect_image meldog_perception_core)

install(TARGETS meldog_perception_core person_detector_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS camera_listener detect_image
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models
  FILES_MATCHING PATTERN "*.onnx")

ament_package()
