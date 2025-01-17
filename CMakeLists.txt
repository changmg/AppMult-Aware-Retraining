# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.16)

# Set the project name
project(simulator)

# Add sub directories
add_subdirectory(simulator/abc)
add_subdirectory(simulator)
