add_library(common STATIC
    matrix.cpp
    mnist_loader.cpp
)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
