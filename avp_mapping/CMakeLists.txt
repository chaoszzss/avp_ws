################ 语义地图编译 #####################
cmake_minimum_required(VERSION 2.8.3)
project(avp_mapping)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -Wno-dev")
add_definitions(-DDATASET_PATH="${PROJECT_SOURCE_DIR}/ipm_data/")

# 添加rt库:linux系统的实时库
link_libraries(rt)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf
  cv_bridge
  message_filters
)

include(cmake/FindEigen.cmake)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
# find_package(OpenMP REQUIRED)

message(STATUS "Build ros node")
catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    include
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
)

# include_directories(${catkin_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED
    src/pose_interpolation.cc
    src/avp_mapping.cc
    src/map.cc
    src/slot.cc
    src/semantic_downsampler.cc
    src/heatmap_utils.cc
)
target_link_libraries(${PROJECT_NAME} 
        ${OpenCV_LIBS} 
        ${PCL_LIBRARIES}
        ${catkin_LIBRARIES}     
        ${cv_bridge_LIBRARIES}
)

# 添加可执行文件
add_executable(avp_mapping_node 
    src/avp_mapping_node_main.cc
)
target_link_libraries(avp_mapping_node 
    ${PROJECT_NAME}
    pthread 
    # OpenMP::OpenMP_CXX
    )

option(BUILD_MAP_HEATMAP_TOOL "Build offline heatmap comparison tool" OFF)
if(BUILD_MAP_HEATMAP_TOOL)
    add_executable(map_heatmap_tool 
        src/map_heatmap_tool.cc
    )
    target_link_libraries(map_heatmap_tool
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    )
endif()

option(BUILD_TOPOLOGY_SVG_TOOL "Build topology svg exporter" ON)
if(BUILD_TOPOLOGY_SVG_TOOL)
    add_executable(topology_svg_tool
        src/topology_svg_tool.cc
    )
    target_link_libraries(topology_svg_tool
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    )
endif()

option(BUILD_COVERAGE_METRICS_TOOL "Build coverage metrics exporter" ON)
if(BUILD_COVERAGE_METRICS_TOOL)
    add_executable(coverage_metrics_tool
        src/coverage_metrics_tool.cc
    )
    target_link_libraries(coverage_metrics_tool
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    )
endif()

option(BUILD_STORAGE_METRICS_TOOL "Build storage metrics exporter" ON)
if(BUILD_STORAGE_METRICS_TOOL)
    add_executable(storage_metrics_tool
        src/storage_metrics_tool.cc
    )
    target_link_libraries(storage_metrics_tool
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    )
endif()

option(BUILD_MAP_INDICATOR_TOOL "Build map indicator exporter" ON)
if(BUILD_MAP_INDICATOR_TOOL)
    add_executable(map_indicator_tool
        src/map_indicator_tool.cc
    )
    target_link_libraries(map_indicator_tool
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    )
endif()


################ end #####################


 

################ 几何地图编译 ################
# cmake_minimum_required(VERSION 3.0.2)
# project(avp_mapping)

# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

# find_package(catkin REQUIRED COMPONENTS 
#     roscpp cv_bridge std_msgs sensor_msgs nav_msgs 
#     geometry_msgs message_filters tf_conversions tf
# )

# find_package(Eigen3 REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(PCL REQUIRED)

# catkin_package()

# include_directories(
#     include
#     ${catkin_INCLUDE_DIRS}
#     ${Eigen3_INCLUDE_DIRS}
#     ${OpenCV_INCLUDE_DIRS}
#     ${PCL_INCLUDE_DIRS}
# )

# # 添加库
# add_library(${PROJECT_NAME}
#     src/pose_interpolation.cc
#     src/geometric_mapping.cc
#     src/map.cc
#     src/heatmap_utils.cc
# )

# target_link_libraries(${PROJECT_NAME}
#     ${catkin_LIBRARIES}
#     ${Eigen3_LIBRARIES}
#     ${OpenCV_LIBS}
#     ${PCL_LIBRARIES}
# )

# # 添加可执行文件
# add_executable(geometric_mapping_node src/geometric_mapping_node.cc)
# target_link_libraries(geometric_mapping_node ${PROJECT_NAME})

# add_executable(map_heatmap_tool
#     src/map_heatmap_tool.cc
# )
# target_link_libraries(map_heatmap_tool
#     ${PROJECT_NAME}
#     ${catkin_LIBRARIES}
#     ${OpenCV_LIBS}
#     ${PCL_LIBRARIES}
# )

################ end #####################
