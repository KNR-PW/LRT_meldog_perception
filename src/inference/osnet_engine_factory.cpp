#include <filesystem>
#include <stdexcept>

#include "meldog_perception/inference/i_osnet_engine.hpp"

#ifdef MELDOG_HAS_TENSORRT
#include "meldog_perception/inference/trt_osnet_engine.hpp"
#endif

#ifdef MELDOG_HAS_ONNXRUNTIME
#include "meldog_perception/inference/onnx_osnet_engine.hpp"
#endif

namespace meldog_perception::inference {

    std::unique_ptr<IOsNetEngine> createOsNetEngine(const std::string& networkPath)
    {
        const std::string extension = std::filesystem::path(networkPath).extension().string();

        if (extension == ".engine") {
#ifdef MELDOG_HAS_TENSORRT
            return std::make_unique<TrtOsNetEngine>(networkPath);
#else
            throw std::runtime_error("[createOsNetEngine] " + networkPath + " is a TensorRT engine, but this build has no TensorRT (MELDOG_HAS_TENSORRT=OFF)");
#endif
        }

        if (extension == ".onnx") {
#ifdef MELDOG_HAS_ONNXRUNTIME
            return std::make_unique<OnnxOsNetEngine>(networkPath);
#else
            throw std::runtime_error("[createOsNetEngine] " + networkPath + " is an ONNX model, but this build has no ONNX Runtime (MELDOG_HAS_ONNXRUNTIME=OFF)");
#endif
        }

        throw std::invalid_argument("[createOsNetEngine] Unknown model type '" + extension + "', expected .engine or .onnx: " + networkPath);
    }
}
