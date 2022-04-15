cmake_minimum_required(VERSION 3.17)
project(reframe_atoms_test)
set(CMAKE_CXX_STANDARD 14)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${MAIN_PROJECT_DIR}/test/atoms/lib/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/atoms ${MAIN_PROJECT_DIR}/test/atoms/lib/buildAtoms)
include_directories(${MAIN_PROJECT_DIR}/src)

#### Add individual tests below ####
add_executable(TestAtomFieldDataSpherical TestAtomFieldDataSpherical.cpp)
target_link_libraries(TestAtomFieldDataSpherical PUBLIC dft-efe-atoms)
set_target_properties(TestAtomFieldDataSpherical PROPERTIES OUTPUT_NAME "TestAtomFieldDataSpherical")

add_executable(TestAtomSphericalData TestAtomSphericalData.cpp)
target_link_libraries(TestAtomSphericalData PUBLIC dft-efe-atoms)
set_target_properties(TestAtomSphericalData PROPERTIES OUTPUT_NAME "TestAtomSphericalData")
