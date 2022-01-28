if (NOT TARGET dft-efe-basis)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Add source files here
set(DFT-EFE-BASIS-SOURCES
	TriangulationCellDealii.cpp
	TriangulationDealiiSerial.cpp
	LinearCellMappingDealii.cpp
	ParentToChildCellsManagerDealii.cpp)

add_library(dft-efe-basis SHARED ${DFT-EFE-BASIS-SOURCES})

include_directories(../)

find_package(deal.II 9.3.0 REQUIRED HINTS ${DEALII_PATH})
target_include_directories(dft-efe-basis PUBLIC ${DEAL_II_INCLUDE_DIRS})
IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	target_link_libraries (dft-efe-basis PUBLIC ${DEAL_II_LIBRARIES_RELEASE})
ELSE()
	target_link_libraries (dft-efe-basis PUBLIC ${DEAL_II_LIBRARIES_DEBUG})
ENDIF()


if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
		target_include_directories(dft-efe-basis PUBLIC ${MPI_CXX_INCLUDE_PATH})
		set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
	endif()
endif()

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()

target_link_libraries(dft-efe-basis PUBLIC dft-efe-utils ${DFTEFE_MPI_LIBRARIES})
endif()
