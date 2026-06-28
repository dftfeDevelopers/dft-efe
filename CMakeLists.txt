cmake_minimum_required(VERSION 3.17)
project(dft-efe-proj VERSION 0.1)
add_executable(dft-efe src/main.cpp)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	find_package(MPI REQUIRED)
  if (ENABLE_MPI_DEVICE_AWARE)
    add_compile_definitions(DFTEFE_WITH_DEVICE_AWARE_MPI)
    MESSAGE(STATUS "DFTEFE with GPU Aware MPI")
  endif()
	include_directories(${MPI_CXX_INCLUDE_PATH})
	set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
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

# if(ENABLE_CUDA)
# 	add_compile_definitions(DFTEFE_WITH_DEVICE)
# 	add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
# 	enable_language(CUDA)
# endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR}/cmake)

target_link_libraries(dft-efe PUBLIC ${DFTEFE_MPI_LIBRARIES})

add_subdirectory(src/utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
add_subdirectory(src/linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
add_subdirectory(src/basis ${CMAKE_CURRENT_BINARY_DIR}/buildBasis)
add_subdirectory(src/quadrature ${CMAKE_CURRENT_BINARY_DIR}/buildQuadrature)
add_subdirectory(src/atoms ${CMAKE_CURRENT_BINARY_DIR}/buildAtoms)
add_subdirectory(src/electrostatics ${CMAKE_CURRENT_BINARY_DIR}/buildElectrostatics)
add_subdirectory(src/ksdft ${CMAKE_CURRENT_BINARY_DIR}/buildKsdft)
target_link_libraries(dft-efe PUBLIC dft-efe-utils dft-efe-linalg dft-efe-basis dft-efe-quadrature dft-efe-atoms dft-efe-electrostatics dft-efe-ksdft)
include_directories(src)

# Build documentation
option(BUILD_DOCS "Build documentation (requires doxygen and sphinx)" OFF)
if(BUILD_DOCS)
	find_package(Doxygen)
	find_package(Sphinx)
	if(${Doxygen_FOUND} AND ${Sphinx_FOUND})
		add_subdirectory("docs")
	else()
		message(WARNING "Doxygen and/or Sphinx not found -- unable to build documentation.")
	endif()
endif()

