#pragma once

#include <memory>

#include <madrona/python.hpp>

namespace Cartpole {

class Manager {
public:
    struct Config {
        int gpuID;
        uint32_t numWorlds;
        bool debugCompile;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::GPUTensor resetTensor() const; // Bool or Int32
    MADRONA_IMPORT madrona::py::GPUTensor actionTensor() const; // Bool or Int32
    MADRONA_IMPORT madrona::py::GPUTensor stateTensor() const; // Vec4 Float32
    MADRONA_IMPORT madrona::py::GPUTensor rewardTensor() const; // Float32

    MADRONA_IMPORT madrona::py::GPUTensor worldIDTensor() const; // Float32


private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}
