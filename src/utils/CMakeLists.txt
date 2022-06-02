if (NOT TARGET dft-efe-utils)
if(${CMAKE_CXX_STANDARD})
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(ENABLE_CUDA)
	enable_language(CUDA)
	add_compile_definitions(DFTEFE_WITH_DEVICE)
	add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
endif()

# Add source files here
set(DFT-EFE-UTILS-SOURCES
	DeviceAPICalls.cu
	DeviceUtils.cpp
	MemoryManager.cpp
	Exceptions.cpp
	MemoryTransfer.cpp
	ExpModX.cpp
	LogModX.cpp
	MPIRequestersNBX.cpp
	MPIPatternP2P.cpp
  MPICommunicatorP2PKernels.cpp
  MPICommunicatorP2PKernels.cu
  MPIErrorCodeHandler.cpp)

add_library(dft-efe-utils SHARED ${DFT-EFE-UTILS-SOURCES})

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
	endif()
	target_include_directories(dft-efe-utils PUBLIC ${MPI_CXX_INCLUDE_PATH})
	set(DFTEFE_UTILS_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})

endif()


include_directories(../)

target_link_libraries(dft-efe-utils PUBLIC ${DFTEFE_UTILS_MPI_LIBRARIES})
endif()
