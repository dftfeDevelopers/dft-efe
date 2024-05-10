if (NOT TARGET dft-efe-ksdft)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Add source files here
set(DFT-EFE-KSDFT-SOURCES
	Defaults.cpp
	FractionalOccupancyFunction.cpp
	KohnShamOperatorContextFE.cpp
	KineticFE.cpp
	ElectrostaticAllElectronFE.cpp
	KohnShamEigenSolver.cpp)

add_library(dft-efe-ksdft SHARED ${DFT-EFE-KSDFT-SOURCES})

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()

if (NOT TARGET dft-efe-basis)
  add_subdirectory(../basis ${CMAKE_CURRENT_BINARY_DIR}/buildBasis)
endif()

if (NOT TARGET dft-efe-linalg)
 add_subdirectory(../linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
endif()

if (NOT TARGET dft-efe-quadrature)
 add_subdirectory(../quadrature ${CMAKE_CURRENT_BINARY_DIR}/buildQuadrature)
endif()

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
		target_include_directories(dft-efe-ksdft PUBLIC ${MPI_CXX_INCLUDE_PATH})
		set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
	endif()
endif()

include_directories(../)
target_link_libraries(dft-efe-ksdft PUBLIC dft-efe-electrostatics dft-efe-utils dft-efe-basis dft-efe-linalg dft-efe-quadrature ${DFTEFE_MPI_LIBRARIES})
endif()
