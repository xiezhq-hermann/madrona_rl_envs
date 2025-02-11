#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

using namespace madrona;
using namespace madrona::py;

namespace Cartpole {

using CPUExecutor =
    TaskGraphExecutor<Engine, Sim, Config, WorldInit, RendererInitStub>;

struct Manager::Impl {
    Config cfg;
    EpisodeManager *episodeMgr;

    static inline Impl * init(const Config &cfg);

    inline Impl(const Config &c, EpisodeManager *episode_mgr)
        : cfg(c),
          episodeMgr(episode_mgr)
    {}

    inline virtual ~Impl() {};
    virtual void run() = 0;
    virtual Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                Span<const int64_t> dims) = 0;
};

struct Manager::CPUImpl final : public Manager::Impl {
    CPUExecutor mwCPU;

    inline CPUImpl(const Config &cfg,
                   const Cartpole::Config &app_cfg,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits,
                   uint32_t num_exported_buffers)
        : Impl(cfg, episode_mgr),
          mwCPU(ThreadPoolExecutor::Config {
                  .numWorlds = cfg.numWorlds,
                  .renderWidth = 0,
                  .renderHeight = 0,
                  .numExportedBuffers = num_exported_buffers,
                  .cameraMode = ThreadPoolExecutor::CameraMode::None,
                  .renderGPUID = 0,
              },
              app_cfg,
              world_inits)
    {}

    inline virtual ~CPUImpl() final
    {
        free(episodeMgr);
    }

    inline virtual void run() final { mwCPU.run(); }

    virtual inline Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = mwCPU.getExported(slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

struct Manager::GPUImpl final : public Manager::Impl {
    MWCudaExecutor mwGPU;

    inline GPUImpl(const Config &cfg,
                   const Cartpole::Config &app_cfg,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits,
                   uint32_t num_exported_buffers)
        : Impl(cfg, episode_mgr),
          mwGPU({
                  .worldInitPtr = world_inits,
                  .numWorldInitBytes = sizeof(WorldInit),
                  .userConfigPtr = (void *)&app_cfg,
                  .numUserConfigBytes = sizeof(Cartpole::Config),
                  .numWorldDataBytes = sizeof(Sim),
                  .worldDataAlignment = alignof(Sim),
                  .numWorlds = cfg.numWorlds,
                  .numExportedBuffers = num_exported_buffers, 
                  .gpuID = (uint32_t)cfg.gpuID,
                  .cameraMode = StateConfig::CameraMode::None,
                  .renderWidth = 0,
                  .renderHeight = 0,
              }, {
                  "",
                  { CARTPOLE_SRC_LIST },
                  { CARTPOLE_COMPILE_FLAGS },
                  cfg.debugCompile ? CompileConfig::OptMode::Debug :
                      CompileConfig::OptMode::LTO,
                  CompileConfig::Executor::TaskGraph,
              })
    {}

    inline virtual ~GPUImpl() final
    {
        REQ_CUDA(cudaFree(episodeMgr));
    }

    inline virtual void run() final { mwGPU.run(); }
    virtual inline Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = mwGPU.getExported(slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};

Manager::Impl * Manager::Impl::init(const Config &cfg)
{
    EpisodeManager *episode_mgr;

    if (cfg.execMode == ExecMode::CPU ) {
        episode_mgr = (EpisodeManager *)malloc(sizeof(EpisodeManager));
        memset(episode_mgr, 0, sizeof(EpisodeManager));
    } else {
        episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));

        // Set the current episode count to 0
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));
    }

    HeapArray<WorldInit> world_inits(cfg.numWorlds);

    Cartpole::Config app_cfg {};

    for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr
        };
    }

    // Increase this number before exporting more tensors
    uint32_t num_exported_buffers = 5;

    if (cfg.execMode == ExecMode::CPU) {
        return new CPUImpl(cfg, app_cfg, episode_mgr, world_inits.data(),
            num_exported_buffers);
    } else {
        return new GPUImpl(cfg, app_cfg,
            episode_mgr, world_inits.data(), num_exported_buffers);
    }
}

MADRONA_EXPORT Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

MADRONA_EXPORT Manager::~Manager() {}

MADRONA_EXPORT void Manager::step()
{
    impl_->run();
}

MADRONA_EXPORT Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(0, Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds, 1});
}

MADRONA_EXPORT Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(1, Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds, 1, 1});
}

MADRONA_EXPORT Tensor Manager::stateTensor() const
{
    return impl_->exportTensor(2, Tensor::ElementType::Float32,
                               {impl_->cfg.numWorlds, 1, 4});
}

MADRONA_EXPORT Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(3, Tensor::ElementType::Float32,
                               {impl_->cfg.numWorlds, 1, 1});
}

MADRONA_EXPORT Tensor Manager::worldIDTensor() const
{
    return impl_->exportTensor(4, Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds, 1, 1});
}

}
