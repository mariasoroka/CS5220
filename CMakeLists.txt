cmake_minimum_required(VERSION 3.14)
project(parallel_bvh LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()



set(SOURCES src/obj_loader.h 
            src/aabb.h 
            src/bvh_builder.h 
            src/builder.h 
            src/vector.h 
            src/obj_loader.h
            src/triangle.h

            src/main.cpp 
            src/obj_loader.cpp 
            src/aabb.cpp 
            src/bvh_builder.cpp 
            src/vector.cpp
            src/obj_loader.cpp
            src/triangle.cpp)

add_executable(build_tree ${SOURCES})

target_include_directories(build_tree PRIVATE ${CMAKE_SOURCE_DIR}/eigen ${CMAKE_SOURCE_DIR}/libigl/include)



# Copy the job scripts
configure_file(job-serial job-serial COPYONLY)
