if(${CMAKE_CXX_STANDARD})
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

message (" DFTEFE_BLAS_LIBRARIES = ${DFTEFE_BLAS_LIBRARIES}")
message (" DFTEFE_SCALAPACK_LIBRARIES = ${DFTEFE_SCALAPACK_LIBRARIES}")


if ("${DFTEFE_BLAS_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFTEFE_BLAS libraries not specified. Please specify -DDFTEFE_BLAS_LIBRARIES in command line")
endif ()


if ("${DFTEFE_SCALAPACK_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFTEFE_SCALAPACK libraries not specified. Please specify -DDFTEFE_SCALAPACK_LIBRARIES in command line")
endif ()

if(ENABLE_CUDA)
    enable_language(CUDA)
if(NOT DEFINED DDFTEFE_WITH_DEVICE)
    ADD_DEFINITIONS(-DDFTEFE_WITH_DEVICE)
    ADD_DEFINITIONS(-DDFTEFE_WITH_DEVICE_CUDA)
endif()
endif()

if (NOT MPI_FOUND)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
endif()

# Add source files here
set(DFT-EFE-LINALG-SOURCES
        VectorKernels.cu
        Vector.cpp
        VectorKernels.cpp)

add_library(dft-efe-linalg SHARED ${DFT-EFE-LINALG-SOURCES})

#FIXME: can we use target_include_directories instead of include_directories
#target_include_directories(dft-efe-linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../)

if(ENABLE_CUDA)
   set_target_properties(dft-efe-linalg PROPERTIES
                          CUDA_STANDARD 14
                          CUDA_STANDARD_REQUIRED ON)
   find_package(CUDAToolkit REQUIRED)
   target_link_libraries(dft-efe-linalg PRIVATE CUDA::cublas)
endif()

if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  if((NOT TARGET ${CMAKE_CURRENT_BINARY_DIR}/buildUtils/dft-efe-utils))
    add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
  endif()
endif()  


target_link_libraries(dft-efe-linalg PRIVATE dft-efe-utils ${DFTEFE_MPI_LIBRARIES} ${DFTEFE_BLAS_LIBRARIES} ${DFTEFE_SCALAPACK_LIBRARIES})
