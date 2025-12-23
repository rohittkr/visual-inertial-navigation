import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "data/video.mp4"
cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Detect good features
prev_pts = cv2.goodFeaturesToTrack(
    prev_gray,
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=10
)

trajectory = []
x, y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None
    )

    good_old = prev_pts[status == 1]
    good_new = next_pts[status == 1]

    # Average motion
    motion = np.mean(good_new - good_old, axis=0)

    x += motion[0]
    y += motion[1]
    trajectory.append((x, y))

    # Draw tracked points
    for pt in good_new:
        cv2.circle(frame, tuple(pt.astype(int)), 2, (0, 255, 0), -1)

    # Resize frame for display
    display_frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Feature Tracking", display_frame)


    prev_gray = gray
    prev_pts = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# Plot trajectory
trajectory = np.array(trajectory)
plt.plot(trajectory[:,0], trajectory[:,1])
plt.title("Vision-based Trajectory")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("results/vision_trajectory.png", dpi=300)
plt.show()

