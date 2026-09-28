find_package(Threads REQUIRED)

add_executable(stage2_benchmark benchmark.cpp matrix_fast.cpp)
target_link_libraries(stage2_benchmark PRIVATE common Threads::Threads)
