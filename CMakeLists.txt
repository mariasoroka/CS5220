cmake_minimum_required(VERSION 3.18)
project(parallel_bvh LANGUAGES C CXX CUDA)

# Add the source code
file(GLOB SOURCE_FILES CONFIGURE_DEPENDS
    src/*.h
    src/*.cpp
    src/*.cu)

add_executable(build_tree ${SOURCE_FILES})
target_compile_features(build_tree PUBLIC cxx_std_20)
target_compile_features(build_tree PUBLIC cuda_std_17)
set_target_properties(build_tree PROPERTIES CUDA_ARCHITECTURES 80)
set_target_properties(build_tree PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(build_tree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_include_directories(build_tree PRIVATE
    ${CMAKE_SOURCE_DIR}/eigen 
    ${CMAKE_SOURCE_DIR}/libigl/include)

# Copy the job scripts
configure_file(job-serial job-serial COPYONLY)
