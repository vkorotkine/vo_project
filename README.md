## A Small Project for Visual Odometry
A small visual odometry system. 
RGBD data is used to localize the camera displacement. 


https://github.com/user-attachments/assets/035ecf82-e8e1-45af-9921-79c0764787da


The system can run in pure PnP-type localization, where landmarks are initialized from depth with PnP used to localize to them, as well as in a bundle-adjustement tracking mode. With bundle adjustement enabled,
reprojection and depth-based stereo factors are used to optimize trajectory over a window of frames. 
The system is tested on the first 500 frames of the ```rgbd_dataset_freiburg1_xyz``` TUM sequence.
Resultant APEs are given by 

```
APE Rotation Error (deg): 2.24
APE Position Error (m): 0.10
```
with bundle adjustement disabled. 

With bundle adjustment enabled,
```
APE Rotation Error (deg): 2.20
APE Position Error (m): 0.07
```


# Docker Container
```
cd docker && docker compose up
```

In VS Code, install Dev Containers,
then Attach to Running Container. 

# Building
```
cd /home/vio_ws/src/vio_project
cmake --preset debug           # configure
cmake --build --preset debug   # build
```
or
```
cd /home/vio_ws/src/vio_project
cmake --preset release           # configure
cmake --build --preset release   # build
```

# Run
```
/home/vio_ws/src/vio_project/build/cpp/app/main
```

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
cd /home/vio_ws/src/vio_project && cmake --build --preset debug && /home/vio_ws/src/vio_project/build/cpp/app/main && /opt/eval-venv/bin/python3 /home/vio_ws/src/vio_project/python/eval/compare_trajectories.py
```
should give 
```
APE Rotation Error (deg): 2.24
APE Position Error (m): 0.10
```
with bundle adjustement disabled. 

With bundle adjustment enabled,
```
APE Rotation Error (deg): 2.20
APE Position Error (m): 0.07
```

# Visualization with Rerun
Rerun (system Python, numpy>=2) does not play well with the host display from inside
Docker. The log is saved and then visualized on the host.
On host machine, 
```
cd /home/vassili/projects/vo_project/output/ && rerun my_log.rrd
```
Replace ``cd`` command with codebase location. 

# AI Usage Disclaimer
This project is mainly intended to learn visual odometry basics. Therefore, AI is not used to directly write
actual code. (Except some of the more tedious Lie group functions. My bad.)
However, I used Claude a good amount for understanding both the general approaches and getting code syntax.
That being said, all code in this repo was written by me, with the exception of peripheral stuff. Dockerfile, some CMakeLists lines, and CMakePresets, venvs. 

# Remaining Future Work
- Write a yaml loader for the config. 
- Work on visualization to draw correspondences. Add video here. 
