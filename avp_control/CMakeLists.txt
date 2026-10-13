cmake_minimum_required(VERSION 2.8.3)
project(avp_control)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-std=c++17 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
add_definitions(-DMESH_PATH="${PROJECT_SOURCE_DIR}/meshes/car.dae")
add_definitions(-DDATASET_PATH="${PROJECT_SOURCE_DIR}/data/")

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf lgsvl_msgs)


include(cmake/FindEigen.cmake)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
)

catkin_package()

add_executable(avp_control
        src/avp_control_node.cpp
        src/ros_viewer.cpp
        src/KinematicModel.cpp
        src/controller.cpp
)

target_link_libraries(avp_control ${catkin_LIBRARIES} ${OpenCV_LIBS})