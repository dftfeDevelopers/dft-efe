cmake_minimum_required(VERSION 3.17)
project(dft-efe VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(MPI REQUIRED)

find_package(BLAS REQUIRED)

message( STATUS BLAS found: ${BLAS_LIBRARIES} )

include_directories(${MPI_CXX_INCLUDE_PATH})
set(MPI_LIBS ${MPI_CXX_LIBRARIES})

set(LIBS ${MPI_LIBS})

add_executable(dft-efe src/main.cpp)
target_link_libraries(dft-efe ${LIBS})
