project(BrainClustering)
cmake_minimum_required(VERSION 2.8.3)
set(BOOST_REQUESTED_VERSION 1.71)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags         -Winline \ -Wno-error=inline \
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -fopenmp \
        -Wall \
        -Wextra \
        -Wpedantic \
        -fdiagnostics-color=auto")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        -O3 \
        -funroll-loops \
        -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
        -O0 \
        -ggdb")

FIND_PACKAGE(PythonInterp 3.8)
FIND_PACKAGE(PythonLibs 3 REQUIRED)

if (PYTHONINTERP_FOUND)
    if (UNIX AND NOT APPLE)
        FIND_PACKAGE(Boost COMPONENTS python numpy serialization)
    else ()
        FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} numpy serialization)
    endif ()
else ()
    message("Python not found")
endif ()

message(STATUS "BUILD TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
LINK_LIBRARIES(
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}) # Deprecated but so convenient!

include_directories(util)

add_custom_target(main ALL)

ADD_SUBDIRECTORY(cluster)
ADD_SUBDIRECTORY(map)