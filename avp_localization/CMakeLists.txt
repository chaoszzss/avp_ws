cmake_minimum_required(VERSION 2.8.3)
project(avp_localization)

set(CMAKE_CXX_FLAGS "-std=c++17 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
add_definitions(-DDATASET_PATH="${PROJECT_SOURCE_DIR}/data/")

find_package(Boost REQUIRED COMPONENTS date_time iostreams)
include_directories(${Boost_INCLUDE_DIRS})
find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf roslib)

include_directories(${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

include(cmake/FindEigen.cmake)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_library(lib_map SHARED
    src/map.cpp
    src/ekf.cpp
    src/frame.cpp
    src/slot.cpp
    src/avp_localization.cpp
    src/map_viewer_ros.cpp
)
target_link_libraries(lib_map ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(avp_localization
        src/avp_localization_node.cpp
)

target_link_libraries(avp_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} lib_map ${Boost_LIBRARIES})
