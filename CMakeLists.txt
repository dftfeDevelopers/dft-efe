cmake_minimum_required(VERSION 3.17)
project(dft-efe VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

message (" BLAS_DIR = ${BLAS_DIR}")
message (" BLAS_LIBRARIES = ${BLAS_LIBRARIES}")

if (${BLAS_DIR}  STREQUAL "")
	message (FATAL_ERROR  " Blas directory not specified. Please specify -DBLAS_DIR in command line")
endif ()

if (${BLAS_LIBRARIES}  STREQUAL "")
	message (FATAL_ERROR  " BLAS libraries not specified. Please specify -DBLAS_LIBRARIES in command line")
endif ()

link_directories(${BLAS_DIR})


include_directories(${MPI_CXX_INCLUDE_PATH})
set(MPI_LIBS ${MPI_CXX_LIBRARIES})

set(LIBS ${MPI_LIBS} ${BLAS_LIBRARIES})


add_executable(dft-efe src/main.cpp)
target_link_libraries(dft-efe ${LIBS})
