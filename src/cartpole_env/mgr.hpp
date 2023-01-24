#pragma once

#include <memory>

#include <madrona/python.hpp>

namespace Cartpole {

class Manager {
public:
    enum class ExecMode {
        CPU,
        CUDA,
    };
    
    struct Config {
        ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        bool debugCompile;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::Tensor resetTensor() const; // Bool or Int32
    MADRONA_IMPORT madrona::py::Tensor actionTensor() const; // Bool or Int32
    MADRONA_IMPORT madrona::py::Tensor stateTensor() const; // Vec4 Float32
    MADRONA_IMPORT madrona::py::Tensor rewardTensor() const; // Float32

    MADRONA_IMPORT madrona::py::Tensor worldIDTensor() const; // Float32


private:
    struct Impl;
    struct CPUImpl;
    struct GPUImpl;

    std::unique_ptr<Impl> impl_;
};

}
