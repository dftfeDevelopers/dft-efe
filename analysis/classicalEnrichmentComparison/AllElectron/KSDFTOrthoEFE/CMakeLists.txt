cmake_minimum_required(VERSION 3.20)
project(poisson_problem_comprison)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

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

list(APPEND CMAKE_MODULE_PATH ${MAIN_PROJECT_DIR}/cmake)

include_directories(${MAIN_PROJECT_DIR}/src)
add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${MAIN_PROJECT_DIR}/test/basis/lib/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/basis ${MAIN_PROJECT_DIR}/test/basis/lib/buildBasis)
add_subdirectory(${MAIN_PROJECT_DIR}/src/atoms ${MAIN_PROJECT_DIR}/test/basis/lib/buildAtoms)
add_subdirectory(${MAIN_PROJECT_DIR}/src/quadrature ${MAIN_PROJECT_DIR}/test/basis/lib/buildQuadrature)
add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${MAIN_PROJECT_DIR}/test/linearAlgebra/lib/buildLinearAlgebra)
add_subdirectory(${MAIN_PROJECT_DIR}/src/electrostatics ${MAIN_PROJECT_DIR}/test/electrostatics/lib/electrostatics)
add_subdirectory(${MAIN_PROJECT_DIR}/src/ksdft ${MAIN_PROJECT_DIR}/test/ksdft/lib/ksdft)

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
add_executable(TestKohnShamDft ${MAIN_PROJECT_DIR}/analysis/classicalEnrichmentComparison/AllElectron/KSDFTOrthoEFE/TestKohnShamDft.cpp)
target_link_libraries(TestKohnShamDft PUBLIC dft-efe-basis dft-efe-electrostatics dft-efe-ksdft dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
endif()
