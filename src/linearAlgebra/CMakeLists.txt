if (NOT TARGET dft-efe-linalg)
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


# Add source files here
set(DFT-EFE-LINALG-SOURCES
		DeviceBlasLapackTemplates.cu
		DeviceLAContextsSingleton.cu
		VectorKernels.cu
		Vector.cpp
		VectorKernels.cpp
		SerialDenseMatrix.cpp
		DistributedDenseMatrix.cpp
		DeviceLAContextsSingleton.cpp)

add_library(dft-efe-linalg SHARED ${DFT-EFE-LINALG-SOURCES})

if(ENABLE_CUDA)
    enable_language(CUDA)
	add_compile_definitions(DFTEFE_WITH_DEVICE)
	add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
endif()

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
		target_include_directories(dft-efe-linalg PUBLIC ${MPI_CXX_INCLUDE_PATH})
		set(DFTEFE_LINEARALGEBRA_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
	endif()
endif()

include_directories(../)

if(ENABLE_CUDA)
   set_target_properties(dft-efe-linalg PROPERTIES
                          CUDA_STANDARD 14
                          CUDA_STANDARD_REQUIRED ON)
   find_package(CUDAToolkit REQUIRED)
   target_link_libraries(dft-efe-linalg PRIVATE CUDA::cublas)
endif()

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()


target_link_libraries(dft-efe-linalg PUBLIC dft-efe-utils ${DFTEFE_LINEARALGEBRA_MPI_LIBRARIES} ${DFTEFE_BLAS_LIBRARIES} ${DFTEFE_SCALAPACK_LIBRARIES})
endif()
