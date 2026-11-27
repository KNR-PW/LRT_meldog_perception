#pragma once

#include <mutex>
#include <iostream>

#include <NvInfer.h>

namespace meldog_perception::inference {

    class TrtLogger final : public nvinfer1::ILogger
    {

    public:

        explicit TrtLogger(Severity minSeverity = Severity::kWARNING) : _minSeverity(minSeverity) {}

        void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
        {
            if (severity > _minSeverity) {
                return;
            }

            std::lock_guard<std::mutex> lock(_mutex);

            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                    std::cerr << "[TRT INTERNAL_ERROR] " << msg << '\n';
                    break;
                case Severity::kERROR:
                    std::cerr << "[TRT ERROR] " << msg << '\n';
                    break;
                case Severity::kWARNING:
                    std::cout << "[TRT WARNING] " << msg << '\n';
                    break;
                case Severity::kINFO:
                    std::cout << "[TRT INFO] " << msg << '\n';
                    break;
                case Severity::kVERBOSE:
                    std::cout << "[TRT VERBOSE] " << msg << '\n';
                    break;
            }
        }

    private:
        Severity _minSeverity;
        std::mutex _mutex;
    };
}
