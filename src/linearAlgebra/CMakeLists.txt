message (" DFTEFE_BLAS_LIBRARIES = ${DFTEFE_BLAS_LIBRARIES}")
message (" DFTEFE_SCALAPACK_LIBRARIES = ${DFTEFE_SCALAPACK_LIBRARIES}")


if ("${DFTEFE_BLAS_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFTEFE_BLAS libraries not specified. Please specify -DDFTEFE_BLAS_LIBRARIES in command line")
endif ()


if ("${DFTEFE_SCALAPACK_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFTEFE_SCALAPACK libraries not specified. Please specify -DDFTEFE_SCALAPACK_LIBRARIES in command line")
endif ()

# Add source files here
set(DFT-EFE-LINALG-SOURCES
        Vector.t.cpp)

add_library(dft-efe-linalg SHARED ${DFT-EFE-LINALG-SOURCES} )
target_include_directories(dft-efe-linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_CUDA)
    set_target_properties(dft-efe-linalg PROPERTIES
                          CUDA_STANDARD 14
                          CUDA_STANDARD_REQUIRED ON)
   find_package(CUDAToolkit REQUIRED)
   target_link_libraries(dft-efe-linalg PRIVATE CUDA::cublas)
endif()

target_link_libraries(dft-efe-linalg PRIVATE dft-efe-utils ${DFTEFE_MPI_LIBRARIES} ${DFTEFE_BLAS_LIBRARIES} ${DFTEFE_SCALAPACK_LIBRARIES})
