cmake_minimum_required(VERSION 2.8.3)
project(BP_Net)

SET(CMAKE_CXX_FLAGS "-std=c++0x")
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})

include_directories(
  ./BpNet
  ./Function
  ./OpenGL
  )

add_executable(bp_node main.cpp BpNet/bpNet.cpp Function/function.cpp OpenGL/OpenGL.cpp)
target_link_libraries(bp_node -lrt -lpthread ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})


