# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.16)

# Set the project name
project(simulator)

# Set top directory
set(TOP_DIR ${PROJECT_SOURCE_DIR}/..)

# Collect source files
aux_source_directory(${PROJECT_SOURCE_DIR}/src sources)

# Add include paths
include_directories(${TOP_DIR}/abc/src)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Add definitions
add_definitions(-DLIN64 -g -Wall -O3)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${TOP_DIR})

# Add an executable
add_executable(${PROJECT_NAME}.out ${sources})
target_link_libraries(${PROJECT_NAME}.out libabc -lm -ldl -rdynamic -lreadline -ltermcap -lpthread -lstdc++ -lrt -lstdc++fs -lboost_timer -lboost_system)
