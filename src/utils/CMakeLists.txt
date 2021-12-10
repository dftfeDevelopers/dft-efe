if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
if(NOT DEFINED DDFTEFE_WITH_DEVICE)
    ADD_DEFINITIONS(-DDFTEFE_WITH_DEVICE)
    ADD_DEFINITIONS(-DDFTEFE_WITH_DEVICE_CUDA)

endif()
endif()

# Add source files here
set(DFT-EFE-UTILS-SOURCES
        DeviceAPICalls.cu
        DeviceUtils.cpp
        MemoryManager.cpp
        Exceptions.cpp
        MemoryTransfer.cpp
	LogModX.cpp)

add_library(dft-efe-utils SHARED ${DFT-EFE-UTILS-SOURCES})

#target_include_directories(dft-efe-utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../)
