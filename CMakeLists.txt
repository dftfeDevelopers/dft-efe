cmake_minimum_required(VERSION 3.17)
project(dft-efe-proj VERSION 0.1)
add_executable(dft-efe src/main.cpp)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})

if(ENABLE_CUDA)
    ADD_DEFINITIONS(-DDFTEFE_WITH_DEVICE)
    ADD_DEFINITIONS(-DDFTEFE_WITH_DEVICE_CUDA)    
    enable_language(CUDA)
endif()

find_package(deal.II 9.3.0 REQUIRED)
include_directories(${DEAL_II_INCLUDE_DIRS})

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  target_link_libraries (dft-efe PUBLIC ${DEAL_II_LIBRARIES_RELEASE})
ELSE()
  target_link_libraries (dft-efe PUBLIC ${DEAL_II_LIBRARIES_DEBUG})
ENDIF()



target_link_libraries(dft-efe PUBLIC ${DFTEFE_MPI_LIBRARIES})

add_subdirectory(src/utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
add_subdirectory(src/linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
target_link_libraries(dft-efe PUBLIC dft-efe-utils dft-efe-linalg)
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

