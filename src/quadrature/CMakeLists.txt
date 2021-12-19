message( "cmake include path = ${CMAKE_INCLUDE_PATH}")

find_package(deal.II 9.3.0 REQUIRED)
include_directories(${DEAL_II_INCLUDE_DIRS})

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


if (NOT MPI_FOUND)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
endif()

# Add source files here
set(DFT-EFE-QUADRATURE-SOURCES
	QuadratureRule.cpp
	CellQuadratureContainer.cpp
        QuadratureRuleGauss.cpp
	QuadratureRuleAdaptive.cpp
	Integrate.cpp)

add_library(dft-efe-quadrature SHARED ${DFT-EFE-QUADRATURE-SOURCES})

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()

if (NOT TARGET dft-efe-basis)
  add_subdirectory(../basis ${CMAKE_CURRENT_BINARY_DIR}/buildBasis)
endif()

include_directories(../)
target_link_libraries(dft-efe-quadrature PUBLIC dft-efe-utils dft-efe-basis ${DFTEFE_MPI_LIBRARIES})
