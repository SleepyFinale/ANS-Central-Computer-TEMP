# ANS-Central-Computer
All Files Relating to Central Computer for ESET 419/420 Capstone

## Start of Each Terminal
source /opt/ros/humble/setup.bash

export TURTLEBOT3_MODEL=burger

## Terminal 0: Robot
ssh ubuntu@172.20.10.8

ros2 launch turtlebot3_bringup robot.launch.py

## Terminal 1: Cartographer
ros2 launch turtlebot3_cartographer cartographer.launch.py configuration_basename:=turtlebot3_lds_2d_improved.lua

## Terminal 2: Nav2
./check_tf_ready.sh

ros2 launch nav2_bringup navigation_launch.py params_file:=/home/schen08/nav2_params_improved.yaml

## Terminal 3: Explorer
source /home/schen08/ros2_ws/install/setup.bash

ros2 run custom_explorer explorer
