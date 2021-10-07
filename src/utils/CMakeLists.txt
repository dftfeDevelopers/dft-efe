
# Add source files here
set(DFT-EFE-UTILS-SOURCES
    MemoryManager.cpp
    DeviceUtils.cpp)

add_library(dft-efe-utils SHARED  ${DFT-EFE-UTILS-SOURCES} )
target_include_directories(dft-efe-utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(dft-efe-utils PRIVATE ${DFTEFE_MPI_LIBRARIES})

if(ENABLE_CUDA)
    set_target_properties(dft-efe-utils PROPERTIES
                          CUDA_STANDARD 14
                          CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(dft-efe-utils PRIVATE CUDA::cudart)
endif()
