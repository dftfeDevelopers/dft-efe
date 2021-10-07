# Add source files here
set(DFT-EFE-UTILS-SOURCES
        DeviceAPICalls.cu
        DeviceUtils.cpp)



add_library(dft-efe-utils SHARED ${DFT-EFE-UTILS-SOURCES})
#find_package(CUDA)

target_include_directories(dft-efe-utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})


if (ENABLE_CUDA)
    set_target_properties(dft-efe-utils PROPERTIES
            CUDA_STANDARD 14
            CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(dft-efe-utils PRIVATE CUDA::cudart)
endif ()

#IF (ENABLE_CUDA)
#    set_source_files_properties(${DFT-EFE-UTILS-SOURCES} PROPERTIES LANGUAGE CUDA)
#ENDIF ()
