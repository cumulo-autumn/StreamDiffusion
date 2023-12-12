import os
import sys
from typing import Literal

import ffmpeg
import fire
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import re
import subprocess

from wrapper import StreamDiffusionWrapper


def get_gpu_power_usage():
    try:
        # nvidia-smiコマンドを実行
        smi_output = subprocess.check_output(["nvidia-smi", "-q", "-d", "POWER"], encoding="utf-8")

        # 正規表現で消費電力を探す
        power_usage_match = re.search(r"Power Draw\s+:\s+(\d+\.\d+) W", smi_output)
        if power_usage_match:
            power_usage = float(power_usage_match.group(1))
            return power_usage
        else:
            return "消費電力の情報が見つかりません"
    except Exception as e:
        return str(e)


def main(
    input: str,
    output: str = "output",
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    prompt: str = "Girl with panda ears wearing a hood",
    scale: float = 1.0,
    acceleration: Literal["none", "xformers", "sfast", "tensorrt"] = "xformers",
):
    video_info = read_video(input)
    video = video_info[0][-22:-2] / 255
    fps = video_info[2]["video_fps"]
    width = int(video.shape[1] * scale)
    height = int(video.shape[2] * scale)

    stream = StreamDiffusionWrapper(
        model_id=model_id,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        is_drawing=False,
        mode="img2img",
        output_type="pt",
        enable_similar_image_filter=True,
        similar_image_filter_threshold=0.99,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    video_result = torch.zeros(video.shape[0], width, height, 3)
    video_result_wo_filter = torch.zeros(video.shape[0], width, height, 3)

    num_runs = 10
    all_probs = []
    all_utilization_rec = []
    all_gpu_power_usages = []
    all_utilization_wo_filter = []
    all_gpu_power_usages_wo_filter = []

    for run in range(num_runs):
        probs = []
        utilization_rec = []
        gpu_power_usages = []
        utilization_wo_filter = []
        gpu_power_usages_wo_filter = []

        stream.stream.enable_similar_image_filter(threshold=0.98)

        for _ in range(stream.batch_size):
            stream(image=video[0].permute(2, 0, 1))

        for i in tqdm(range(video.shape[0])):
            output_image, skip_prob = stream(video[i].permute(2, 0, 1))
            probs.append(skip_prob)
            video_result[i] = output_image.permute(1, 2, 0)
            utilization_rec.append(torch.cuda.utilization())
            gpu_power_usages.append(get_gpu_power_usage())

        stream.stream.disable_similar_image_filter()

        for _ in range(stream.batch_size):
            stream(image=video[0].permute(2, 0, 1))

        for i in tqdm(range(video.shape[0])):
            output_image, skip_prob = stream(video[i].permute(2, 0, 1))
            video_result_wo_filter[i] = output_image.permute(1, 2, 0)
            utilization_wo_filter.append(torch.cuda.utilization())
            gpu_power_usages_wo_filter.append(get_gpu_power_usage())

        all_probs.append(probs)
        all_utilization_rec.append(utilization_rec)
        all_gpu_power_usages.append(gpu_power_usages)
        all_utilization_wo_filter.append(utilization_wo_filter)
        all_gpu_power_usages_wo_filter.append(gpu_power_usages_wo_filter)

    # 各リストの平均を計算
    avg_probs = [sum(col) / len(col) for col in zip(*all_probs)]
    avg_utilization_rec = [sum(col) / len(col) for col in zip(*all_utilization_rec)]
    avg_gpu_power_usages = [sum(col) / len(col) for col in zip(*all_gpu_power_usages)]
    avg_utilization_wo_filter = [sum(col) / len(col) for col in zip(*all_utilization_wo_filter)]
    avg_gpu_power_usages_wo_filter = [sum(col) / len(col) for col in zip(*all_gpu_power_usages_wo_filter)]

    video_result = video_result * 255
    video_result_wo_filter = video_result_wo_filter * 255

    write_video(f"{output}_w_filter.mp4", video_result, fps=fps)
    write_video(f"{output}_wo_filter.mp4", video_result_wo_filter, fps=fps)

    video_cat = torch.cat([video_result, video_result_wo_filter], dim=2)
    write_video(f"{output}_cat.mp4", video_cat, fps=fps)

    # save fig of gpu usage
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.ticker import MaxNLocator

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(bottom=0.15)

    # 選択するフレーム間隔を決定（例：全フレーム数の1/10）
    frame_interval = 1
    # プロットのための画像サンプルを選択
    sample_frames = [video[i].numpy() for i in range(0, len(video), frame_interval)]
    # 各サンプル画像をサブプロットとして追加
    for i, frame in enumerate(sample_frames):
        imagebox = OffsetImage(frame, zoom=0.09)
        ab = AnnotationBbox(imagebox, (0, 0), xybox=(i * 1.0, -0.30), xycoords="data", pad=0.0, frameon=False)
        ax = plt.gca()
        ax.add_artist(ab)

    ax1.set_ylabel("Mean of Skip Probability", color="tab:red", fontsize=14)
    ax1.plot(np.arange(len(avg_probs)), avg_probs, color="tab:red", label="Skip Probability", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper right")
    # Creating a second y-axis for skip probability
    ax2 = ax1.twinx()
    # Plotting the GPU utilization
    ax1.set_xlabel("Frame", fontsize=14)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylabel("GPU Utilization (%)", color="tab:blue", fontsize=14)
    ax2.plot(np.arange(len(avg_utilization_rec)), avg_utilization_rec, color="tab:blue", label="Mean of GPU Utilization with filter", linewidth=3)
    ax2.plot(
        np.arange(len(avg_utilization_wo_filter)),
        avg_utilization_wo_filter,
        color="tab:orange",
        label="Mean of GPU Utilization without filter",
        linewidth=2,
    )
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.legend(loc="upper left")
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

    # Title and layout adjustments
    # plt.title("Mean of GPU Processor Utilization and Skip Probability", fontsize=16)
    fig.tight_layout()
    # Save the figure to a file
    plt.savefig(os.path.join("gpu_utilization.png"), dpi=300)

    # out put the data as csv
    import pandas as pd

    df = pd.DataFrame(
        {
            "skip_prob": avg_probs,
            "utilization": avg_utilization_rec,
            "utilization_wo_filter": avg_utilization_wo_filter,
            "gpu_power_usages": avg_gpu_power_usages,
            "gpu_power_usages_wo_filter": avg_gpu_power_usages_wo_filter,
        }
    )
    df.to_csv("gpu_utilization.csv")


if __name__ == "__main__":
    fire.Fire(main)
