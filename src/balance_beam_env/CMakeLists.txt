set(SIMULATOR_SRCS
  sim.hpp sim.cpp
)

madrona_build_compile_defns(
  OUT_TARGET
  balance_example_gpu_srcs
  SOURCES_DEFN
  BALANCE_SRC_LIST
  FLAGS_DEFN
  BALANCE_COMPILE_FLAGS 
  SRCS
  ${SIMULATOR_SRCS}
)

add_library(balance_example_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(balance_example_mgr PRIVATE
    madrona_python_utils
    madrona_mw_gpu
    madrona_common
    madrona_importer
    balance_example_gpu_srcs
)

target_compile_definitions(balance_example_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_balance_example_python
    bindings.cpp
)

target_link_libraries(madrona_balance_example_python PRIVATE
    madrona_hdrs
    balance_example_mgr
)
