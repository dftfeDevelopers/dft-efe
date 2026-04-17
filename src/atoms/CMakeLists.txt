if (NOT TARGET dft-efe-atoms)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

IF (ENABLE_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
      enable_language(CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
      enable_language(HIP)
    ENDIF()
ENDIF()

# Add source files here
set(DFT-EFE-ATOMS-SOURCES
    SphericalDataNumerical.cpp
    Defaults.cpp
    SphericalHarmonicFunctions.cpp
    SmoothCutoffFunctions.cpp
    AtomSphericalDataEnrichment.cpp
    AtomSphericalDataEnrichmentJSON.cpp
    AtomSphericalDataContainer.cpp
    BoostAutoDiff.cpp
    AtomSevereFunction.cpp
    AtomSphericalDataPSP.cpp
    SphericalDataMixed.cpp
    SphericalDataAnalytical.cpp
    AtomSphericalDataAnalytical.cpp
    AtomTCIASpline.cpp
   )

IF (ENABLE_GPU)
    SET(DFT-EFE-ATOMS-DEVICE-SOURCES
        SmoothCutoffFunctionsDeviceKernels.cpp
        SphericalHarmonicFunctionsDeviceKernels.cpp
        SphericalDataNumericalDeviceKernels.cpp
        AtomSevereFunctionDeviceKernels.cpp
        )
    IF ("${GPU_LANG}" STREQUAL "cuda")
        set_source_files_properties(${DFT-EFE-ATOMS-DEVICE-SOURCES} PROPERTIES LANGUAGE CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
        set_source_files_properties(${DFT-EFE-ATOMS-DEVICE-SOURCES} PROPERTIES LANGUAGE HIP)
        IF ("${GPU_VENDOR}" STREQUAL "amd")
            ADD_DEFINITIONS(-D__HIP_PLATFORM_AMD__)
        ELSEIF ("${GPU_VENDOR}" STREQUAL "nvidia")
            ADD_DEFINITIONS(-D__HIP_PLATFORM_NVIDIA__)
        ENDIF()
    ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
        set_source_files_properties(${DFT-EFE-ATOMS-DEVICE-SOURCES} PROPERTIES COMPILE_FLAGS ${CMAKE_SYCL_FLAGS})
    ENDIF()
    LIST(APPEND DFT-EFE-ATOMS-SOURCES ${DFT-EFE-ATOMS-DEVICE-SOURCES})
ENDIF()

add_library(dft-efe-atoms SHARED ${DFT-EFE-ATOMS-SOURCES})

target_include_directories(dft-efe-atoms PUBLIC ${LIBXML_PATH})
target_link_libraries(dft-efe-atoms PUBLIC ${LIBXML_LIBRARIES})

target_include_directories(dft-efe-atoms PRIVATE ${BOOST_DIR}/include)
#target_link_directories(dft-efe-atoms PUBLIC ${BOOST_DIR}/lib64)
target_link_directories(dft-efe-atoms PUBLIC ${BOOST_DIR}/lib)

include_directories(../)

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_Declare(
  simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
  GIT_TAG v4.3.0
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(simdjson)

target_link_libraries(dft-efe-atoms PUBLIC dft-efe-utils simdjson)
endif()
