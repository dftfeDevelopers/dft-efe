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
	QuadratureRuleAdaptive.cpp)

add_library(dft-efe-quadrature SHARED ${DFT-EFE-QUADRATURE-SOURCES})

#FIXME: can we use target_include_directories instead of include_directories
#target_include_directories(dft-efe-linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../)



target_link_libraries(dft-efe-quadrature PUBLIC dft-efe-utils dft-efe-basis ${DFTEFE_MPI_LIBRARIES})
