# Visual–Inertial Navigation and Sensor Fusion Pipeline

## Overview
This project implements a **basic Visual–Inertial Odometry (VIO) pipeline**
using a monocular camera and IMU data.  
The system estimates relative motion in **simulated GPS-denied environments**
by combining vision-based motion estimation with inertial sensing.

The goal of this project is to demonstrate **sensor fusion fundamentals**
rather than achieving production-level accuracy.

---

## System Pipeline
1. Camera video input
2. Feature detection using Shi–Tomasi corner detector
3. Feature tracking using Lucas–Kanade Optical Flow
4. IMU motion estimation via acceleration integration
5. Vision + IMU fusion using weighted filtering
6. Trajectory visualization and comparison

---

## Technologies Used
- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

vio_project/
├── vision.py # Vision-only odometry
├── imu_fusion.py # IMU + vision fusion
├── generate_imu.py # IMU data simulator
├── README.md
├── data/
│ ├── video.mp4
│ └── imu.csv
└── results/
└── vio_comparison.png
