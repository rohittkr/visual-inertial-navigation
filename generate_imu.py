import pandas as pd
import numpy as np

t = np.arange(0, 10, 0.01)

imu = pd.DataFrame({
    "timestamp": t,
    "ax": np.random.normal(0, 0.1, len(t)),
    "ay": np.random.normal(0, 0.1, len(t)),
    "az": np.ones(len(t)) * 9.8,  # gravity
    "gx": np.random.normal(0, 0.01, len(t)),
    "gy": np.random.normal(0, 0.01, len(t)),
    "gz": np.random.normal(0, 0.01, len(t))
})

imu.to_csv("data/imu.csv", index=False)

print("imu.csv generated")
