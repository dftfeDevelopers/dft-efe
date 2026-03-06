if (NOT TARGET dft-efe-utils)
if(${CMAKE_CXX_STANDARD})
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

# if(ENABLE_CUDA)
# 	enable_language(CUDA)
# 	add_compile_definitions(DFTEFE_WITH_DEVICE)
# 	add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
# endif()

# Add source files here
set(DFT-EFE-UTILS-SOURCES
	#DeviceAPICalls.cu
	DeviceUtils.cpp
	#DeviceComplexUtils.cu
	DiscontiguousDataOperations.cpp
	#DiscontiguousDataOperations.cu
  MemoryManager.cpp
	Exceptions.cpp
	NumberUtils.cpp
	MemoryTransfer.cpp
	ExpModX.cpp
	LogModX.cpp
	MathFunctions.cpp
	MathConstants.cpp
	StringOperations.cpp
	ConditionalOStream.cpp
	MPITypes.cpp
	MPIWrapper.cpp
	MPIRequestersNBX.cpp
	MPIPatternP2P.cpp
	Defaults.cpp
	RandNumGen.cpp
  #MPICommunicatorP2PKernels.cpp
  #MPICommunicatorP2PKernels.cu
  MPIErrorCodeHandler.cpp
  Spline.cpp
  ScalarZeroFunctionReal.cpp
  SmearChargeDensityFunction.cpp
  SmearChargePotentialFunction.cpp
  PointChargePotentialFunction.cpp
  Timer.cpp
  Profiler.cpp)

IF (ENABLE_GPU)
  IF ("${GPU_LANG}" STREQUAL "cuda")
  SET(DFT-EFE-UTILS-DEVICE-SOURCES
      DeviceUtils.cpp
      DeviceAPICalls.cu.cpp
      DiscontiguousDataOperationsKernelsDevice.cpp
    )
  ELSEIF ("${GPU_LANG}" STREQUAL "hip")
  SET(DFT-EFE-UTILS-DEVICE-SOURCES
      DeviceUtils.cpp
      DeviceAPICalls.hip.cpp
      DiscontiguousDataOperationsKernelsDevice.cpp
    )
  ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
  SET(DFT-EFE-UTILS-DEVICE-SOURCES
      DeviceUtils.cpp
      DeviceAPICalls.sycl.cpp
      DiscontiguousDataOperationsKernelsDevice.cpp
    )
  ENDIF()
ENDIF()

IF (ENABLE_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
       set_source_files_properties(${DFT-EFE-UTILS-DEVICE-SOURCES} PROPERTIES LANGUAGE CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
       set_source_files_properties(${DFT-EFE-UTILS-DEVICE-SOURCES} PROPERTIES LANGUAGE HIP)
       IF ("${GPU_VENDOR}" STREQUAL "amd")
          ADD_DEFINITIONS(-D__HIP_PLATFORM_AMD__)
       ELSEIF ("${GPU_VENDOR}" STREQUAL "nvidia")
          ADD_DEFINITIONS(-D__HIP_PLATFORM_NVIDIA__)
       ENDIF()
    ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
      set_source_files_properties(${DFT-EFE-UTILS-DEVICE-SOURCES} PROPERTIES COMPILE_FLAGS ${CMAKE_SYCL_FLAGS})
    ENDIF()

  LIST(APPEND DFT-EFE-UTILS-SOURCES ${DFT-EFE-UTILS-DEVICE-SOURCES})
ENDIF()

add_library(dft-efe-utils SHARED ${DFT-EFE-UTILS-SOURCES})

IF (ENABLE_GPU)
  IF ("${GPU_LANG}" STREQUAL "cuda")
    MESSAGE(STATUS "DFTEFE with GPU LANG CUDA")
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(dft-efe-utils PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(dft-efe-utils PUBLIC CUDA::cudart)
    target_link_libraries(dft-efe-utils PUBLIC CUDA::cublas)
  ELSEIF ("${GPU_LANG}" STREQUAL "hip")
    MESSAGE(STATUS "DFTEFE with GPU LANG HIP")
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_HIP)
    find_package(hipblas)
    target_link_libraries(dft-efe-utils PUBLIC roc::hipblas)
  ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
    MESSAGE(STATUS "DFTEFE with GPU LANG SYCL")
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_SYCL)
    find_program(MPI_BASE_CXX_COMPILER "icpx")
    get_filename_component(_REALPATH_SYCL_COMPILER "${MPI_BASE_CXX_COMPILER}" REALPATH)
    get_filename_component(SYCL_BIN_DIR "${_REALPATH_SYCL_COMPILER}" DIRECTORY)
    get_filename_component(SYCL_PACKAGE_DIR "${SYCL_BIN_DIR}" DIRECTORY CACHE)
    find_file(SYCL_INCLUDE_DIR
    NAMES
    include
    HINTS
    ${SYCL_PACKAGE_DIR}
    NO_DEFAULT_PATH)
    target_include_directories(dft-efe-utils PUBLIC ${SYCL_INCLUDE_DIR})
    target_include_directories(dft-efe-utils PUBLIC ${SYCL_INCLUDE_DIR}/sycl)
    set(MKL_INTERFACE lp64)
    set(ENABLE_SCALAPACK ON)
    set(MKL_MPI mpich)
    find_PACKAGE(IntelSYCL REQUIRED)
    find_package(MKL REQUIRED)
    target_link_libraries(dft-efe-utils PUBLIC MKL::MKL)
    get_target_property(MKL_SYCL_INCLUDE_DIRS MKL::MKL_SYCL INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(dft-efe-utils PUBLIC ${MKL_SYCL_INCLUDE_DIRS})
    get_target_property(mkl_sycl_includes MKL::MKL_SYCL INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(dft-efe-utils PUBLIC ${mkl_sycl_includes})
    get_target_property(mkl_sycl_libs MKL::MKL_SYCL INTERFACE_LINK_LIBRARIES)
    target_link_libraries(dft-efe-utils PUBLIC ${mkl_sycl_libs})
    set_target_properties(dft-efe-utils PROPERTIES LINK_FLAGS ${CMAKE_SYCL_FLAGS})
  ENDIF()
ENDIF()

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
	endif()
  if (ENABLE_MPI_DEVICE_AWARE)
    add_compile_definitions(DFTEFE_WITH_DEVICE_AWARE_MPI)
    MESSAGE(STATUS "DFTEFE with GPU Aware MPI")
  endif()
	target_include_directories(dft-efe-utils PUBLIC ${MPI_CXX_INCLUDE_PATH})
	set(DFTEFE_UTILS_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})

endif()

target_include_directories(dft-efe-utils PRIVATE ${BOOST_DIR}/include)
target_link_directories(dft-efe-utils PUBLIC ${BOOST_DIR}/lib64)
target_link_directories(dft-efe-utils PUBLIC ${BOOST_DIR}/lib)

include_directories(../)

target_link_libraries(dft-efe-utils PUBLIC boost_regex ${DFTEFE_UTILS_MPI_LIBRARIES})
endif()
