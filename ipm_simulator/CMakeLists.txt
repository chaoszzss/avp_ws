cmake_minimum_required(VERSION 3.0.2)
project(ipm_simulator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 查找 ROS 和 OpenCV/Eigen 依赖
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
)
find_package(Boost REQUIRED COMPONENTS 
  filesystem 
  system
  )


if(NOT OpenCV_DIR)
  set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
endif()
find_package(OpenCV 4 REQUIRED)
# 强制设置OpenMP
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/usr/lib/x86_64-linux-gnu/libgomp.so.1")
set(OpenMP_CXX_FOUND TRUE)



# 声明 catkin 包
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ipm_simulator
  CATKIN_DEPENDS 
    roscpp 
    sensor_msgs 
    cv_bridge 
    image_transport 
    message_filters
)

# 包含头文件和库路径
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 编译主节点
add_executable(ipm_node
  src/ipm_node.cpp
  src/ipm_fun.cc
)

add_executable(ipm_node_label
  src/ipm_node_label.cpp
  src/ipm_fun.cc
)

add_executable(ipm_node_raw
  src/ipm_node_raw.cpp
  src/ipm_fun.cc
)

add_executable(ipm_weight_export
  src/ipm_weight_export.cpp
  src/ipm_fun.cc
)

# 链接库
target_link_libraries(ipm_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenMP_omp_LIBRARY}
  Boost::filesystem
)

target_link_libraries(ipm_node_label
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenMP_omp_LIBRARY}
  Boost::filesystem
)

target_link_libraries(ipm_node_raw
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenMP_omp_LIBRARY}
  Boost::filesystem
)

target_link_libraries(ipm_weight_export
  ${OpenCV_LIBRARIES}
  ${OpenMP_omp_LIBRARY}
  Boost::filesystem
)
