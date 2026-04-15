# Source
set(CORE_SOURCES
        core/Host.cpp
        core/Timed.cpp
        core/Callback.cpp
)

set(SPIKING_SOURCES
        spiking/Network.cpp
        spiking/Membrane.cpp
)

set(FUNCTIONAL_SOURCES
)

set(SIM_SOURCES
        Simulator.cpp
        SimulationWorker.cpp
        SimulationLogger.cpp
)


file(GLOB CUDA_SOURCES "spiking/cuda/*.cu")


# Source groups
source_group("core"           FILES ${CORE_SOURCES})
source_group("core/meta"      FILES core/meta/type_list.h
              core/meta/type_pack.h
              core/meta/con_expander.h
              core/meta/con_maker.h
              core/meta/con_process.h
              core/meta/rule_expander.h)
source_group("spiking"        FILES ${NETWORK_SOURCES})
source_group("cuda"           FILES ${CUDA_SOURCES})
source_group("simulator"      FILES ${SIM_SOURCES})

# Libraries
add_library(braincel STATIC
        ${CORE_SOURCES}
        ${SPIKING_SOURCES}
        ${FUNCTIONAL_SOURCES}
        ${CUDA_SOURCES}
        ${SIM_SOURCES}
)

set_target_properties(braincel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES native
)

target_include_directories(braincel
        PUBLIC  ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(braincel
        PUBLIC  nlohmann_json::nlohmann_json
        Python3::Python
        PRIVATE CUDA::cudart
        CUDA::curand
)