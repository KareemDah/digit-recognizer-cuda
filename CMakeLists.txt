cmake_minimum_required(VERSION 3.15)
project(digit_recognizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(common)
add_subdirectory(stage1_inference)

# Further stage subdirectories are added here as we build them.
