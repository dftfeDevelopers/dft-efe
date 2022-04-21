cmake_minimum_required(VERSION 3.17)
project(dft-efe-proj VERSION 0.1)
add_executable(dft-efe src/main.cpp)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	find_package(MPI REQUIRED)
	include_directories(${MPI_CXX_INCLUDE_PATH})
	set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
endif()

if(ENABLE_CUDA)
	add_compile_definitions(DFTEFE_WITH_DEVICE)
	add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
	enable_language(CUDA)
endif()

target_link_libraries(dft-efe PUBLIC ${DFTEFE_MPI_LIBRARIES})

add_subdirectory(src/utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
add_subdirectory(src/linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
add_subdirectory(src/basis ${CMAKE_CURRENT_BINARY_DIR}/buildBasis)
add_subdirectory(src/quadrature ${CMAKE_CURRENT_BINARY_DIR}/buildQuadrature)
add_subdirectory(src/atoms ${CMAKE_CURRENT_BINARY_DIR}/buildAtoms)
target_link_libraries(dft-efe PUBLIC dft-efe-utils dft-efe-linalg dft-efe-basis dft-efe-quadrature dft-efe-atoms)
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

