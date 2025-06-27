cmake_minimum_required(VERSION 3.10)
project(CNN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src src/lib)
file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/lib/*.cpp"
)

add_executable(cnn_main ${SOURCES})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
