cmake_minimum_required(VERSION 3.5)
project(multi_agent_sac)

find_package(catkin REQUIRED COMPONENTS
  roslib
  roslint
)

catkin_package(
 CATKIN_DEPENDS roslib roslint
)

# linting
roslint_python()
roslint_add_test()
