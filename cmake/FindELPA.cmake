# FindELPA.cmake
#
# Finds the ELPA library
#
# This will define the following variables
#
#    ELPA_FOUND
#    ELPA_INCLUDE_DIRS
#    ELPA_LIBRARIES
#
# and the following imported targets
#
#     ELPA::ELPA
#
# Author: David M. Rogers <predictivestatmech@gmail.com>

# Allow user to set ELPA_DIR
set(${ELPA_DIR} "" CACHE PATH "Path to ELPA installation")

#Make sure pkg-config uses CMAKE_PREFIX_PATH
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)

if(EXISTS "${ELPA_DIR}")
    message(STATUS "The elpa path '${ELPA_DIR}' exists.")
    # Add ELPA_DIR to search paths if provided
    list(APPEND CMAKE_PREFIX_PATH "${ELPA_DIR}")
    list(APPEND ENV{PKG_CONFIG_PATH} "${ELPA_DIR}/lib/pkgconfig")
endif()

find_package(PkgConfig)
#set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
foreach(pkg elpa_openmp elpa) # prioritize elpa_openmp
    pkg_search_module(PC_ELPA ${pkg})
    if(PC_ELPA_FOUND)
        break()
    endif()
    if(PC_ELPA_FOUND)
        break()
    endif()
endforeach()

if(ELPA_FIND_REQUIRED AND NOT PC_ELPA_FOUND)
    MESSAGE(FATAL_ERROR "Unable to find ELPA. Try adding dir containing lib/pkgconfig/elpa.pc to -DCMAKE_PREFIX_PATH")
endif()

find_path(ELPA_INCLUDE_DIR
    NAMES elpa/elpa.h elpa/elpa_constants.h
    PATHS ${PC_ELPA_INCLUDE_DIRS}
)
find_library(ELPA_LIBRARIES
    #    NAMES elpa elpa_openmp
    NAMES ${PC_ELPA_LIBRARIES}
    PATHS ${PC_ELPA_LIBRARY_DIRS}
    DOC "elpa libraries list"
)
MESSAGE(STATUS "Using ELPA_INCLUDE_DIR = ${ELPA_INCLUDE_DIR}")
MESSAGE(STATUS "Using ELPA_LIBRARIES = ${ELPA_LIBRARIES}")
set(ELPA_VERSION ${PC_ELPA_VERSION})

mark_as_advanced(ELPA_FOUND ELPA_INCLUDE_DIR ELPA_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ELPA
    REQUIRED_VARS ELPA_INCLUDE_DIR ELPA_LIBRARIES
    VERSION_VAR ELPA_VERSION
)

if(ELPA_FOUND)
    set(ELPA_INCLUDE_DIRS ${ELPA_INCLUDE_DIR})
endif()

if(ELPA_FOUND AND NOT TARGET ELPA::ELPA)
    add_library(ELPA::ELPA INTERFACE IMPORTED)
    set_target_properties(ELPA::ELPA PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ELPA_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${ELPA_LIBRARIES}"
        #INTERFACE_COMPILE_FEATURES c_std_99
    )
endif()