cmake_minimum_required(VERSION 3.20)
project(poisson_problem_comprison)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

include_directories(${MAIN_PROJECT_DIR}/src)
add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${MAIN_PROJECT_DIR}/test/basis/lib/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/atoms ${MAIN_PROJECT_DIR}/test/basis/lib/buildAtoms)

add_executable(PSPConverter ${MAIN_PROJECT_DIR}/analysis/PseudoConverter/PSPConverter.cpp)
target_link_libraries(PSPConverter PUBLIC dft-efe-utils dft-efe-atoms)
