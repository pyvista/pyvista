#!/bin/sh
set -x

# sudo apt-get update && sudo apt-get install python-qt4 libgl1-mesa-glx
sudo apt-get install python-qt4 libgl1-mesa-glx
sudo apt-get install -y xvfb
export PYVISTA_VIRTUAL_DISPLAY=True
export PYVISTA_OFF_SCREEN=True
# Debugging commands:
# ls -l /etc/init.d/
# sh -e /etc/init.d/xvfb start
# give xvfb some time to start
sleep 3
set +x
