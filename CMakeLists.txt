cmake_minimum_required(VERSION 3.15)
project(digit_recognizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build -- benchmarks are meaningless without -O2,
# and this is a performance-focused project.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Statically link the MinGW runtime. Without this, the exe depends on
# libstdc++-6.dll/libwinpthread-1.dll being found on PATH at runtime -- on a
# machine with more than one MinGW install (e.g. Git Bash's bundled one),
# the wrong, ABI-incompatible DLL can get picked up and crash std::thread
# code. Static linking also makes the exe portable on its own.
if(MINGW)
    add_link_options(-static)
endif()

add_subdirectory(common)
add_subdirectory(stage1_inference)
add_subdirectory(stage2_fast)

# Further stage subdirectories are added here as we build them.
