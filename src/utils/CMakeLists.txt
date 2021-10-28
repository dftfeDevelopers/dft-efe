# Add source files here
set(DFT-EFE-UTILS-SOURCES
        DeviceAPICalls.cu
        DeviceUtils.cpp
        MemoryManager.cpp
        Exceptions.cpp
        MemoryTransfer.cpp)

add_library(dft-efe-utils SHARED ${DFT-EFE-UTILS-SOURCES})

target_include_directories(dft-efe-utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
