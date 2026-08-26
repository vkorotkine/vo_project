## A Small Project for Visual Odometry
A work in progress to build and evaluate a visual odometry system. 

# SSH Into Docker Container
In VS Code, install Dev Containers,
then Attach to Running Container. 

# Building
cd /home/vio_ws/src/vio_project
cmake --preset debug           # configure
cmake --build --preset debug   # build

# Run
/home/vio_ws/src/vio_project/build/cpp/app/main

# Tests
/home/vio_ws/src/vio_project/build/cpp/tests/unit_tests
