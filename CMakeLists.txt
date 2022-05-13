cmake_minimum_required(VERSION 3.21)
project(Open_Test_Cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(Open_Test_Cpp main.cpp cmake-build-debug/Harris_Detector.h)


set(OpenCV_DIR E:\\Users\\Kevin\\opencv\\mingw_build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Open_Test_Cpp ${OpenCV_LIBS})
