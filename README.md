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
Unit, 
```
/home/vio_ws/src/vio_project/build/cpp/tests/unit_tests
```
Integration, 
```
/home/vio_ws/src/vio_project/build/cpp/tests/integration_tests
```

# Plot
The eval stack (navlie) needs numpy<2, so it lives in an isolated venv.
```
/opt/eval-venv/bin/python3 /home/vio_ws/src/vio_project/python/eval/compare_trajectories.py
```

# All together
```
cmake --build --preset debug
/home/vio_ws/src/vio_project/build/cpp/app/main
/opt/eval-venv/bin/python3 /home/vio_ws/src/vio_project/python/eval/plot_trajectories.py
```

# Visualization with Rerun
Rerun (system Python, numpy>=2) does not play well with the host display from inside
Docker. The log is saved and then visualized on the host.