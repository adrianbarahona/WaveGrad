cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp_export)

set(app test-app)

set(CMAKE_PREFIX_PATH ../libtorch)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(${app} test.cpp wavreader.cpp)
target_link_libraries(${app} "${TORCH_LIBRARIES}" Eigen3::Eigen)
set_property(TARGET ${app} PROPERTY CXX_STANDARD 14)