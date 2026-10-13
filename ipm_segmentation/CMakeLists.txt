cmake_minimum_required(VERSION 3.0.2)
project(ipm_segmentation)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # 链接 OpenMP
endif()

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  image_transport
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    sensor_msgs 
    cv_bridge 
    image_transport
)

# 安装 Python 脚本（设置可执行权限）
install(PROGRAMS
  scripts/ipm_seg.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装模型文件
install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
  USE_SOURCE_PERMISSIONS
)

# # 安装 Python 库到包的 Python 路径（非全局）
# install(DIRECTORY lib/
#   DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}  # 安装到 devel/lib/python3/dist-packages/ipm_segmentation
#   PATTERN "__pycache__" EXCLUDE
# )

# install(DIRECTORY config/
#   DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION} 
#   PATTERN "__pycache__" EXCLUDE
# )

# # 替换以下内容
# install(DIRECTORY lib/
#   DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}  # 安装到 devel/lib/python3/dist-packages/
#   PATTERN "__pycache__" EXCLUDE
# )

# install(DIRECTORY config/
#   DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}
#   PATTERN "__pycache__" EXCLUDE
# )