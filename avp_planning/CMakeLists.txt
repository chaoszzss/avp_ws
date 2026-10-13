cmake_minimum_required(VERSION 2.8.3)
project(avp_planning)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
add_definitions(-DMESH_PATH="${PROJECT_SOURCE_DIR}/meshes/car.dae")
add_definitions(-DDATASET_PATH="${PROJECT_SOURCE_DIR}/data/")

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf)


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

add_executable(avp_planning
        src/avp_planning_node.cpp
        src/avp_map.cpp
        src/slot.cpp
        src/ros_viewer.cpp
        src/planner.cpp
)

target_link_libraries(avp_planning ${catkin_LIBRARIES} ${OpenCV_LIBS})