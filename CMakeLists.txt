cmake_minimum_required(VERSION 3.6)
project(face_demo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(OpenCV_DIR /usr/local/opencv-3.4.5/bulid/) 
find_package(OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES  FaceDetector.h FacePreprocess.h FaceRecognition.h ModelLoader.cpp)

#add_executable(face_demo ${SOURCE_FILES} main.cpp)
#target_link_libraries(face_demo ${OpenCV_LIBS})

add_library(face_python_ext MODULE ${SOURCE_FILES} pythonExt.cpp)
target_link_libraries(face_python_ext ${OpenCV_LIBS} )

add_executable(sample sample.cpp ModelLoader.cpp)
target_link_libraries(sample ${OpenCV_LIBS})