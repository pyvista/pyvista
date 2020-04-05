#!/bin/sh
set -x

# sudo apt-get update && sudo apt-get install python-qt4 libgl1-mesa-glx
sudo apt-get install python-qt4 libgl1-mesa-glx
sudo apt-get install -y xvfb
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=True
which Xvfb
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
# Debugging commands:
# ls -l /etc/init.d/
# sh -e /etc/init.d/xvfb start
# give xvfb some time to start
sleep 3
set +x
