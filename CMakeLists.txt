cmake_minimum_required(VERSION 3.24)
project(opencv_SORT)

set(CMAKE_CXX_STANDARD 14)
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

add_definitions(-DDEBUG=1)

#寻找OpenCV库
FIND_PACKAGE(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV version: ${OpenCV_VERSION}")
MESSAGE(STATUS "       libraries: ${OpenCV_LIBRARY_DIRS}")
MESSAGE(STATUS "       include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


include_directories(sort-c++)
aux_source_directory(${PROJECT_SOURCE_DIR}/sort-c++ SRCX)
add_executable(${PROJECT_NAME} ${SRCX} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})