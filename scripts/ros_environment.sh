#!/usr/bin/env bash
# Source this file from project launchers; keep machine-wide settings unchanged.
PERSON_CALLER_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PERSON_ROBOT_SETUP="${PERSON_ROBOT_SETUP:-/home/rpp/rpp_ws/install/setup.bash}"
if [[ -f "${PERSON_ROBOT_SETUP}" ]]; then
    source "${PERSON_ROBOT_SETUP}"
fi
# starterButton itself does not source ~/.rpprc, unlike the robot driver entry.
# Match the robot's localhost discovery unless the caller explicitly chose it.
export ROS_LOCALHOST_ONLY="${PERSON_CALLER_LOCALHOST_ONLY}"
