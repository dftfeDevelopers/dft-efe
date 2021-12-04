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
set(DFT-EFE-BASIS-SOURCES
	TriangulationCellDealii.cpp
	TriangulationDealiiSerial.cpp
	LinearCellMappingDealii.cpp
	ParentToChildCellsManagerDealii.cpp)

add_library(dft-efe-basis SHARED ${DFT-EFE-BASIS-SOURCES})

#FIXME: can we use target_include_directories instead of include_directories
#target_include_directories(dft-efe-linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../)


#if(NOT TARGET ${CMAKE_CURRENT_BINARY_DIR}/../utils/dft-efe-utils)
#add_subdirectory(${CMAKE_SOURCE_DIR}/../utils ${CMAKE_CURRENT_SOURCE_DIR})
#	add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/../utils/)
#endif()

target_link_libraries(dft-efe-basis PUBLIC dft-efe-utils ${DFTEFE_MPI_LIBRARIES})
#target_link_libraries(test PUBLIC dft-efe-basis dft-efe-utils ${DFTEFE_MPI_LIBRARIES})
