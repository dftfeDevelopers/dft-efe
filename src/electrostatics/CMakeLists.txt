if (NOT TARGET dft-efe-electrostatics)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

IF (ENABLE_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
      enable_language(CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
      enable_language(HIP)
    ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
    ELSE ()
      MESSAGE(FATAL_ERROR "\n"
    "*** Could not find valid GPU_LANG=cuda/hip/sycl ***\n\n"
    )
    ENDIF()
#
#Set GPU compilation flag
#
  add_compile_definitions(DFTEFE_WITH_DEVICE)

  IF ("${GPU_VENDOR}" STREQUAL "nvidia")
    add_compile_definitions(DFTEFE_WITH_DEVICE_NVIDIA)
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_CUDA)
  ELSEIF ("${GPU_VENDOR}" STREQUAL "amd")
    add_compile_definitions(DFTEFE_WITH_DEVICE_AMD)
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_HIP)
  ELSEIF ("${GPU_VENDOR}" STREQUAL "intel")
    add_compile_definitions(DFTEFE_WITH_DEVICE_INTEL)
    add_compile_definitions(DFTEFE_WITH_DEVICE_MKL)
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_SYCL)
  ENDIF()
ENDIF()

# Add source files here
set(DFT-EFE-ELECTROSTATICS-SOURCES
	PoissonLinearSolverFunctionFE.cpp
	LaplaceOperatorContextFE.cpp
	PoissonSolverDealiiMatrixFreeFE.cpp
  MatrixFree.cpp)

IF (ENABLE_GPU)
IF ("${GPU_LANG}" STREQUAL "cuda")
SET(DFT-EFE-ELECTROSTATICS-DEVICE-SOURCES
     MatrixFreeDevice.cu.cpp
     MatrixFreeDevice.cpp
     PoissonSolverDealiiMatrixFreeFEDeviceKernels.cpp
    )
ELSEIF ("${GPU_LANG}" STREQUAL "hip")
SET(DFT-EFE-ELECTROSTATICS-DEVICE-SOURCES
    MatrixFreeDevice.hip.cpp
    MatrixFreeDevice.cpp
    PoissonSolverDealiiMatrixFreeFEDeviceKernels.cpp
    )
ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
SET(DFT-EFE-ELECTROSTATICS-DEVICE-SOURCES
    MatrixFreeDevice.sycl.cpp
    MatrixFreeDevice.cpp
    PoissonSolverDealiiMatrixFreeFEDeviceKernels.cpp
    )
ENDIF()
ENDIF()

IF (ENABLE_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
    set_source_files_properties(${DFT-EFE-ELECTROSTATICS-DEVICE-SOURCES} PROPERTIES LANGUAGE CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
    set_source_files_properties(${DFT-EFE-ELECTROSTATICS-DEVICE-SOURCES} PROPERTIES LANGUAGE HIP)
    IF ("${GPU_VENDOR}" STREQUAL "amd")
        ADD_DEFINITIONS(-D__HIP_PLATFORM_AMD__)
    ELSEIF ("${GPU_VENDOR}" STREQUAL "nvidia")
        ADD_DEFINITIONS(-D__HIP_PLATFORM_NVIDIA__)
    ENDIF()
    ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
    set_source_files_properties(${DFT-EFE-ELECTROSTATICS-DEVICE-SOURCES} PROPERTIES COMPILE_FLAGS ${CMAKE_SYCL_FLAGS})
    ENDIF()

LIST(APPEND DFT-EFE-ELECTROSTATICS-SOURCES ${DFT-EFE-ELECTROSTATICS-DEVICE-SOURCES})
ENDIF()

add_library(dft-efe-electrostatics SHARED ${DFT-EFE-ELECTROSTATICS-SOURCES})

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()

if (NOT TARGET dft-efe-basis)
  add_subdirectory(../basis ${CMAKE_CURRENT_BINARY_DIR}/buildBasis)
endif()

if (NOT TARGET dft-efe-linalg)
 add_subdirectory(../linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
endif()

if (NOT TARGET dft-efe-quadrature)
 add_subdirectory(../quadrature ${CMAKE_CURRENT_BINARY_DIR}/buildQuadrature)
endif()

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
		target_include_directories(dft-efe-electrostatics PUBLIC ${MPI_CXX_INCLUDE_PATH})
		set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
	endif()
endif()

include_directories(../)
target_link_libraries(dft-efe-electrostatics PUBLIC dft-efe-utils dft-efe-basis dft-efe-linalg dft-efe-quadrature ${DFTEFE_MPI_LIBRARIES})
endif()
