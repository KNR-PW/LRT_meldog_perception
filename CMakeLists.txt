cmake_minimum_required(VERSION 3.17)
project(meldog_perception LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(MELDOG_HAS_TENSORRT "Build GPU inference backend (CUDA + TensorRT, .engine models)" OFF)
option(MELDOG_HAS_ONNXRUNTIME "Build ONNX Runtime inference backend (.onnx models)" ON)

if(NOT MELDOG_HAS_TENSORRT AND NOT MELDOG_HAS_ONNXRUNTIME)
  message(FATAL_ERROR "No inference backend enabled. Set -DMELDOG_HAS_ONNXRUNTIME=ON and/or -DMELDOG_HAS_TENSORRT=ON")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

set(MELDOG_BACKENDS "")

if(MELDOG_HAS_TENSORRT)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA not found. Install the CUDA toolkit (or set -DCUDAToolkit_ROOT=<path>), "
                        "or disable this backend with -DMELDOG_HAS_TENSORRT=OFF")
  endif()

  set(TENSORRT_ROOT "" CACHE PATH "TensorRT install prefix (only for tarball installs)")
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include)
  find_library(TENSORRT_LIBRARY nvinfer HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64)
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found (NvInfer.h / libnvinfer). Install it, set -DTENSORRT_ROOT=<prefix>, "
                        "or disable this backend with -DMELDOG_HAS_TENSORRT=OFF")
  endif()

  file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_VERSION_LINES
    REGEX "#define[ \t]+(NV_TENSORRT|TRT)_(MAJOR|MINOR)(_ENTERPRISE)?[ \t]+[0-9]+")
  string(REGEX REPLACE ".*(MAJOR)(_ENTERPRISE)?[ \t]+([0-9]+).*" "\\3" TENSORRT_VERSION_MAJOR "${TENSORRT_VERSION_LINES}")
  string(REGEX REPLACE ".*(MINOR)(_ENTERPRISE)?[ \t]+([0-9]+).*" "\\3" TENSORRT_VERSION_MINOR "${TENSORRT_VERSION_LINES}")
  set(TENSORRT_VERSION "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}")
  if(TENSORRT_VERSION VERSION_LESS 8.5)
    message(FATAL_ERROR "TensorRT ${TENSORRT_VERSION} found in ${TENSORRT_INCLUDE_DIR}, 8.5 or newer is required")
  endif()

  list(APPEND MELDOG_BACKENDS "CUDA ${CUDAToolkit_VERSION} + TensorRT ${TENSORRT_VERSION}")
endif()

if(MELDOG_HAS_ONNXRUNTIME)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime install prefix (e.g. extracted release tarball in /opt/onnxruntime)")
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_ROOT} /opt/onnxruntime PATH_SUFFIXES include include/onnxruntime)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT} /opt/onnxruntime PATH_SUFFIXES lib lib64)
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found (onnxruntime_cxx_api.h / libonnxruntime). Install it, set -DONNXRUNTIME_ROOT=<prefix>, "
                        "or disable this backend with -DMELDOG_HAS_ONNXRUNTIME=OFF")
  endif()

  file(STRINGS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" ONNXRUNTIME_API_LINE REGEX "#define[ \t]+ORT_API_VERSION[ \t]+[0-9]+")
  string(REGEX REPLACE ".*ORT_API_VERSION[ \t]+([0-9]+).*" "\\1" ONNXRUNTIME_API_VERSION "${ONNXRUNTIME_API_LINE}")
  if(ONNXRUNTIME_API_VERSION LESS 14)
    message(FATAL_ERROR "ONNX Runtime 1.${ONNXRUNTIME_API_VERSION} found in ${ONNXRUNTIME_INCLUDE_DIR}, 1.14 or newer is required")
  endif()

  list(APPEND MELDOG_BACKENDS "ONNX Runtime 1.${ONNXRUNTIME_API_VERSION}")
endif()

string(JOIN ", " MELDOG_BACKENDS_STR ${MELDOG_BACKENDS})
message(STATUS "meldog_perception: OpenCV ${OpenCV_VERSION}, inference backends: ${MELDOG_BACKENDS_STR}")

add_library(meldog_perception_core STATIC
  src/target_identifier.cpp
  src/reid_encoder.cpp
  src/inference/osnet_engine_factory.cpp
)
set_target_properties(meldog_perception_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(meldog_perception_core PUBLIC include)
target_link_libraries(meldog_perception_core PUBLIC ${OpenCV_LIBS})

if(MELDOG_HAS_TENSORRT)
  target_sources(meldog_perception_core PRIVATE src/inference/trt_osnet_engine.cpp)
  target_compile_definitions(meldog_perception_core PRIVATE MELDOG_HAS_TENSORRT)
  target_include_directories(meldog_perception_core SYSTEM PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(meldog_perception_core
    PUBLIC  CUDA::cudart
    PRIVATE ${TENSORRT_LIBRARY}
  )
endif()

if(MELDOG_HAS_ONNXRUNTIME)
  target_sources(meldog_perception_core PRIVATE src/inference/onnx_osnet_engine.cpp)
  target_compile_definitions(meldog_perception_core PRIVATE MELDOG_HAS_ONNXRUNTIME)
  target_include_directories(meldog_perception_core SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(meldog_perception_core PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

add_executable(camera_listener src/camera_listener.cpp)
target_include_directories(camera_listener PRIVATE include)
ament_target_dependencies(camera_listener rclcpp sensor_msgs)

install(TARGETS camera_listener
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()