import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.io import read_video, write_png

df = pd.read_csv("gpu_utilization.csv")

video_info = read_video("examples/mov2mov/mabataki2.mp4")
video = video_info[0][-22:-2] / 255

# for i in range(len(video)):
#     write_png((video[i] * 255).permute(2, 0, 1).to(dtype=torch.uint8), f"images/{i}.png")

avg_probs = df["skip_prob"]
avg_utilization_rec = df["utilization"]
avg_utilization_wo_filter = df["utilization_wo_filter"]
avg_gpu_power_usages = df["gpu_power_usages"]
avg_gpu_power_usages_wo_filter = df["gpu_power_usages_wo_filter"]


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MaxNLocator

fig = plt.figure(figsize=(14, 10))
plt.style.use("ggplot")

plt.subplot(3, 1, 1)
plt.axis("off")
frame_interval = 1
sample_frames = [video[i].numpy() for i in range(0, len(video), frame_interval)]


for i, frame in enumerate(sample_frames):
    imagebox = OffsetImage(frame, zoom=0.095)
    ab = AnnotationBbox(imagebox, (1.0, 0), xybox=(i * 0.05 + 0.023, 0.00), xycoords="data", pad=0.0, frameon=False)
    ax = plt.gca()
    ax.add_artist(ab)

plt.subplot(3, 1, 2)

ax = plt.gca()
ax.set_facecolor((0.9, 0.9, 0.9))
ax.grid(color="w", linestyle="-", linewidth=1)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

ax1 = plt.gca()
ax1.set_ylabel("Average Skip Probability", fontsize=14)
ax1.plot(np.arange(len(avg_probs)), avg_probs, color="tab:red", label="Skip Probability", linewidth=2)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)

ax = plt.gca()
ax.set_facecolor((0.9, 0.9, 0.9))
ax.grid(color="w", linestyle="-", linewidth=1)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Creating a second y-axis for skip probability
ax2 = plt.gca()
# Plotting the GPU utilization
ax2.set_ylabel("Average GPU Usage (%)", fontsize=14)
ax2.plot(np.arange(len(avg_utilization_rec)), avg_utilization_rec, color="tab:blue", label="with filter", linewidth=3)
ax2.plot(
    np.arange(len(avg_utilization_wo_filter)),
    avg_utilization_wo_filter,
    color="tab:orange",
    label="without filter",
    linewidth=2,
)
ax2.legend(loc="upper right", fontsize="large")

ax2.set_xlabel("Frame", fontsize=14)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

# ax2.set_ylabel("Mean of GPU Power Usage (W)", color="tab:blue", fontsize=14)
# ax2.plot(np.arange(len(avg_gpu_power_usages)), avg_gpu_power_usages, color="tab:blue", label="GPU Power Usage with filter", linewidth=2)
# ax2.plot(
#     np.arange(len(avg_gpu_power_usages_wo_filter)),
#     avg_gpu_power_usages_wo_filter,
#     color="tab:orange",
#     label="GPU Power Usage without filter",
#     linewidth=2,
# )
# ax2.tick_params(axis="y", labelcolor="tab:blue")
# ax2.legend(loc="upper left")

plt.tight_layout()
# Save the figure to a file
plt.savefig("gpu_utilization.png", dpi=300)

plt.show()


# avg_gpu_power_usagesの平均
print(sum(avg_gpu_power_usages) / len(avg_gpu_power_usages))
# avg_gpu_power_usages_wo_filterの平均
print(sum(avg_gpu_power_usages_wo_filter) / len(avg_gpu_power_usages_wo_filter))
