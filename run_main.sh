#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate piper310

source /opt/ros/humble/setup.bash
source /home/gavin/piper_ws/install/setup.bash

python /home/gavin/single_cam_apriltag/main.py