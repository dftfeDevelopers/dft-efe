cmake_minimum_required(VERSION 3.17)
project(dft-efe VERSION 0.1)
add_executable(dft-efe src/main.cpp)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
set(DFT_EFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})

find_package(deal.II 9.3.0 REQUIRED)
include_directories(${DEAL_II_INCLUDE_DIRS})

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  target_link_libraries (dft-efe ${DEAL_II_LIBRARIES_RELEASE})
ELSE()
  target_link_libraries (dft-efe ${DEAL_II_LIBRARIES_DEBUG})
ENDIF()


message (" DFT_EFE_BLAS_DIR = ${DFT_EFE_BLAS_DIR}")
message (" DFT_EFE_BLAS_LIBRARIES = ${DFT_EFE_BLAS_LIBRARIES}")
message (" DFT_EFE_SCALAPACK_LIBRARIES = ${DFT_EFE_SCALAPACK_LIBRARIES}")
message (" DFT_EFE_SCALAPACK_DIR = ${DFT_EFE_SCALAPACK_DIR}")

if (NOT EXISTS ${DFT_EFE_BLAS_DIR})
	message (FATAL_ERROR  "DFT_EFE_Blas directory not specified. Please specify -DDFT_EFE_BLAS_DIR in command line")
endif ()
	
if ("${DFT_EFE_BLAS_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFT_EFE_BLAS libraries not specified. Please specify -DDFT_EFE_BLAS_LIBRARIES in command line")
endif ()	

if ("${DFT_EFE_SCALAPACK_DIR}" STREQUAL "")
	message (FATAL_ERROR  "DFT_EFE_SCALAPACK directory not specified. Please specify -DDFT_EFE_SCALAPACK_DIR in command line")
endif ()

if ("${DFT_EFE_SCALAPACK_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFT_EFE_SCALAPACK libraries not specified. Please specify -DDFT_EFE_SCALAPACK_LIBRARIES in command line")
endif ()

link_directories(${DFT_EFE_BLAS_DIR} ${DFT_EFE_SCALAPACK_DIR})


set(DFT_EFE_LIBRARIES ${DFT_EFE_MPI_LIBRARIES} ${DFT_EFE_BLAS_LIBRARIES} ${DFT_EFE_SCALAPACK_LIBRARIES})
target_link_libraries(dft-efe ${DFT_EFE_LIBRARIES})		

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
