cmake_minimum_required(VERSION 3.18)
project(parallel_bvh LANGUAGES C CXX CUDA)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

# Add the source code
file(GLOB SOURCE_FILES CONFIGURE_DEPENDS
    src/*.h
    src/*.cpp
    src/*.cu)

add_executable(build_tree ${SOURCE_FILES})
target_compile_features(build_tree PUBLIC cxx_std_20)
target_compile_features(build_tree PUBLIC cuda_std_20)
set_target_properties(build_tree PROPERTIES CUDA_ARCHITECTURES native)
target_include_directories(build_tree PRIVATE 
    ${CMAKE_SOURCE_DIR}/eigen 
    ${CMAKE_SOURCE_DIR}/libigl/include)

# Copy the job scripts
configure_file(job-serial job-serial COPYONLY)
