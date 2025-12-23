import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# LOAD IMU DATA
# -----------------------------
imu = pd.read_csv("data/imu.csv")

dt = 0.01
vel = np.array([0.0, 0.0])
pos = np.array([0.0, 0.0])

imu_traj = []

for _, row in imu.iterrows():
    acc = np.array([row.ax, row.ay])
    vel += acc * dt
    pos += vel * dt
    imu_traj.append(pos.copy())

imu_traj = np.array(imu_traj)

# -----------------------------
# VISION TRAJECTORY (RECOMPUTE)
# -----------------------------
cap = cv2.VideoCapture("data/video.mp4")
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

pts = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 10)

x, y = 0, 0
vision_traj = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)

    good_old = pts[status == 1]
    good_new = next_pts[status == 1]

    motion = np.mean(good_new - good_old, axis=0)

    x += motion[0]
    y += motion[1]
    vision_traj.append([x, y])

    prev_gray = gray
    pts = good_new.reshape(-1, 1, 2)

cap.release()
vision_traj = np.array(vision_traj)

# -----------------------------
# SENSOR FUSION (WEIGHTED)
# -----------------------------
min_len = min(len(imu_traj), len(vision_traj))
alpha = 0.7  # IMU weight

fused_traj = alpha * imu_traj[:min_len] + (1 - alpha) * vision_traj[:min_len]

# -----------------------------
# PLOT RESULTS
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(vision_traj[:,0], vision_traj[:,1], label="Vision Only")
plt.plot(imu_traj[:,0], imu_traj[:,1], label="IMU Only")
plt.plot(fused_traj[:,0], fused_traj[:,1], label="Fused (Vision + IMU)")
plt.legend()
plt.title("Visual–Inertial Navigation Comparison")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("results/vio_comparison.png", dpi=300)
plt.show()

