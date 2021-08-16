cmake_minimum_required(VERSION 3.17)
project(dft-efe)

set(CMAKE_CXX_STANDARD 14)
find_package(MPI REQUIRED)

if (MPI_FOUND)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    set(MPI_LIBS ${MPI_CXX_LIBRARIES})
elseif(MPI_FOUND)
    message("MPI not found")
endif (MPI_FOUND)

set(LIBS ${MPI_LIBS})

add_executable(dft-efe main.cpp)
target_link_libraries(dft-efe ${LIBS})
