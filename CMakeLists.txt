cmake_minimum_required(VERSION 3.25)

if(APPLE)
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "C++ compiler" FORCE)
endif()

project(RealTimeVisionPipeline
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(OpenCV REQUIRED)
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS /opt/homebrew/opt/onnxruntime/include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS /opt/homebrew/opt/onnxruntime/lib
)

add_executable(vision_app
    src/main.cpp
)

target_link_libraries(vision_app
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

target_include_directories(vision_app
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)