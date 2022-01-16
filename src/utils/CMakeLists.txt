if(${CMAKE_CXX_STANDARD})
	set(CMAKE_CXX_STANDARD 14)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(ENABLE_MPI)
	if(NOT DEFINED DDFTEFE_WITH_DEVICE)
		ADD_DEFINITIONS(-DDFTEFE_WITH_MPI)
	endif()
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
		target_include_directories(dft-efe-utils PUBLIC ${MPI_CXX_INCLUDE_PATH})
		set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
	endif()
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
	ExpModX.cpp
	LogModX.cpp
	OptimizedIndexSet.cpp
	MPIRequestersNBX.cpp)

add_library(dft-efe-utils SHARED ${DFT-EFE-UTILS-SOURCES})


include_directories(../)

target_link_libraries(dft-efe-utils PUBLIC ${DFTEFE_MPI_LIBRARIES})
