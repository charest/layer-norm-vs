#------------------------------------------------------------------------------#
# Set the minimum CMake version
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.11)
#cmake_policy(SET CMP0074 NEW)

#------------------------------------------------------------------------------#
# Setup the project
#------------------------------------------------------------------------------#

project(LayerNorm VERSION 0.1.0)

# We need C++ 17
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)

# cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
include(BuildType)

add_executable( layer_norm )

#------------------------------------------------------------------------------#
# OpenMP toolkit
#------------------------------------------------------------------------------#
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(layer_norm PUBLIC OpenMP::OpenMP_CXX)
endif()

#------------------------------------------------------------------------------#
# Hip
#------------------------------------------------------------------------------#

set(HIP_PLATFORM "amd")
find_package(hip REQUIRED)
message(STATUS "**** HIP Configuration:")
message(STATUS "  VERSION: ${hip_VERSION}")
message(STATUS "  INCLUDE: ${hip_INCLUDE_DIRS}")
message(STATUS "  LIBRARIES: ${hip_LIBRARIES}")
message(STATUS "  GPU_TARGETS: ${GPU_TARGETS}")

#find_package(HIP REQUIRED)
#MESSAGE(STATUS "Available AMDGPU Target CPUs: ${AMDGPU_TARGETS}")

target_link_libraries(layer_norm PRIVATE ${hip_LIBRARIES})


#------------------------------------------------------------------------------#
# build library
#------------------------------------------------------------------------------#

add_subdirectory(layer-norm)
