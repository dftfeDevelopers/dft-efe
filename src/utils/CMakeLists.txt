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
	DeviceComplexUtils.cu
        MemoryManager.cpp
	Exceptions.cpp
	NumberUtils.cpp
	MemoryTransfer.cpp
	ExpModX.cpp
	LogModX.cpp
	MathFunctions.cpp
	StringOperations.cpp
	ConditionalOStream.cpp
	MPITypes.cpp
	MPIWrapper.cpp
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
  if (ENABLE_MPI_DEVICE_AWARE)
    add_compile_definitions(DFTEFE_WITH_DEVICE_AWARE_MPI)
  endif()
	target_include_directories(dft-efe-utils PUBLIC ${MPI_CXX_INCLUDE_PATH})
	set(DFTEFE_UTILS_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})

endif()

#target_include_directories(dft-efe-utils PUBLIC ${BOOST_DIR}/include)
#target_link_directories(dft-efe-utils PUBLIC ${BOOST_DIR}/lib64)
#target_link_directories(dft-efe-utils PUBLIC ${BOOST_DIR}/lib)

include_directories(../)

target_link_libraries(dft-efe-utils PUBLIC ${DFTEFE_UTILS_MPI_LIBRARIES})
endif()
