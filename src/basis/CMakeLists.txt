
# Add source files here
set(DFT-EFE-BASIS-SOURCES
        TriangulationCellDealii.cpp
        TriangulationDealiiSerial.cpp)

add_library(dft-efe-basis SHARED ${DFT-EFE-BASIS-SOURCES} )
target_include_directories(dft-efe-basis INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(dft-efe-basis PRIVATE dft-efe-utils ${DFTEFE_MPI_LIBRARIES})
