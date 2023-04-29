if (NOT TARGET dft-efe-atoms)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Add source files here
set(DFT-EFE-ATOMS-SOURCES
    SphericalData.cpp
    SphericalHarmonics.cpp
    AtomSphericalData.cpp
    AtomSphericalDataContainer.cpp
   )

add_library(dft-efe-atoms SHARED ${DFT-EFE-ATOMS-SOURCES})

target_include_directories(dft-efe-atoms PUBLIC ${LIBXML_PATH})
target_link_libraries(dft-efe-atoms PUBLIC ${LIBXML_LIBRARIES})

target_include_directories(dft-efe-atoms PRIVATE ${BOOST_DIR}/include)
#target_link_directories(dft-efe-atoms PUBLIC ${BOOST_DIR}/lib64)
target_link_directories(dft-efe-atoms PUBLIC ${BOOST_DIR}/lib)

include_directories(../)

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()

target_link_libraries(dft-efe-atoms PUBLIC dft-efe-utils)
endif()
