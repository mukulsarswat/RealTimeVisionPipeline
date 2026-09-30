cmake_minimum_required(VERSION 3.16)

project(RealTimeVisionPipeline
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(OpenCV REQUIRED)

add_executable(vision_app
    src/main.cpp
)

target_link_libraries(vision_app
    PRIVATE
    ${OpenCV_LIBS}
)

target_include_directories(vision_app
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)