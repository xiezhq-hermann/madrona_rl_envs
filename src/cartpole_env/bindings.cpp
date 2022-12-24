#include "mgr.hpp"

#include <madrona/macros.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace Cartpole {

NB_MODULE(madrona_cartpole_example_python, m) {
    nb::class_<Manager> (m, "CartpoleSimulator")
        .def("__init__", [](Manager *self,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            bool debug_compile) {
            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .debugCompile = debug_compile,
            });
        }, nb::arg("gpu_id"), nb::arg("num_worlds"),
           nb::arg("debug_compile") = true)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("state_tensor", &Manager::stateTensor)
        .def("reward_tensor", &Manager::rewardTensor)
    ;
}

}
