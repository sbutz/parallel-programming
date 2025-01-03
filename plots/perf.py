#!/usr/bin/env python3

import argparse
import logging
import math
import matplotlib.ticker as ticker
import os
import pandas as pd
import subprocess
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

git_root = (
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, check=True
    )
    .stdout.decode("utf-8")
    .strip()
)

plot_dir = os.path.join(git_root, "plots")


def build_target(folder, target):
    subprocess.run(
        ["make", "-C", os.path.join(git_root, folder), target],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


class NvProfReport:
    def __init__(self, binary, args):
        logger.info(f'Profiling: {binary} {" ".join(args)}')
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "nsys",
                "profile",
                "--stats=true",
                f"--output={tmpdir}/report%n",
                binary,
                *args,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True
            )
            self.report = result.stdout.decode("utf-8").split("\n")

    def get_kernel_execution_time(self, kernel_name):
        lines = [l for l in self.report if f" {kernel_name}(" in l]
        # [0] = Time (%)
        # [1] = Total Time (ns)
        # [2] = Instances
        # [3] = Avg (ns)
        # [4] = Med (ns)
        # [5] = Min (ns)
        # [6] = Max (ns)
        # [7] = StdDev (ns)
        # [8] = Name
        parts = lines[0].split()
        return float(parts[3]) / 1000  # in us

    def get_kernel_launch_time(self):
        lines = [l for l in self.report if "cudaLaunchKernel" in l]
        # [0] = Time (%)
        # [1] = Total Time (ns)
        # [2] = Num Calls
        # [3] = Avg (ns)
        # [4] = Med (ns)
        # [5] = Min (ns)
        # [6] = Max (ns)
        # [7] = StdDev (ns)
        # [8] = Name
        parts = lines[0].split()
        return float(parts[3]) / 1000  # in us

    def get_memcpy_to_device_time(self):
        lines = [l for l in self.report if "[CUDA memcpy Host-to-Device]" in l]
        # [0] = Time (%)
        # [1] = Total Time (ns)
        # [2] = Count
        # [3] = Avg (ns)
        # [4] = Med (ns)
        # [5] = Min (ns)
        # [6] = Max (ns)
        # [7] = StdDev (ns)
        # [8] = Name
        parts = lines[0].split()
        return float(parts[3]) / 1000  # in us

    def get_memcpy_to_host_time(self):
        lines = [l for l in self.report if "[CUDA memcpy Device-to-Host]" in l]
        # [0] = Time (%)
        # [1] = Total Time (ns)
        # [2] = Count
        # [3] = Avg (ns)
        # [4] = Med (ns)
        # [5] = Min (ns)
        # [6] = Max (ns)
        # [7] = StdDev (ns)
        # [8] = Name
        parts = lines[0].split()
        return float(parts[3]) / 1000  # in us


def image_filter():
    # Build binaries
    build_target("01_image_filter", "all")
    image_dir = os.path.join(git_root, "01_image_filter", "images")
    images = [
        os.path.join(image_dir, i)
        for i in ["01_small.jpg", "02_medium.jpg", "03_large.jpg"]
    ]
    binary = os.path.join(git_root, "01_image_filter", "build", "02_blur")
    kernel_name = "Blur"

    # Profile kernels
    d = []
    for image in images:
        for margin in [1, 2, 3]:
            with tempfile.NamedTemporaryFile() as tmpfile:
                report = NvProfReport(binary, [str(margin), image, tmpfile.name])
                d.append(
                    {
                        "image": os.path.basename(image),
                        "margin": margin,
                        "exec_time": report.get_kernel_execution_time(kernel_name),
                        "launch_time": report.get_kernel_launch_time(),
                        "memcpy_to_device_time": report.get_memcpy_to_device_time(),
                        "memcpy_to_host_time": report.get_memcpy_to_host_time(),
                    },
                )
    df = pd.DataFrame(d)

    # Plot execution times
    fig = (
        df.pivot(index="image", columns="margin", values="exec_time")
        .plot(kind="bar")
        .get_figure()
    )
    fig.gca().set_title("Kernel execution times")
    fig.gca().set_ylabel("Time [us]")
    fig.gca().set_yscale("log", base=2)
    fig.gca().set_xticklabels(fig.gca().get_xticklabels(), rotation=0)
    fig.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plot_path = os.path.join(plot_dir, "01_image_filter", "kernel_execution_times.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)

    # Plot memcpy times
    fig = (
        df[df["margin"] == 1]
        .set_index("image")[
            [
                "launch_time",
                "exec_time",
                "memcpy_to_device_time",
                "memcpy_to_host_time",
            ]
        ]
        .plot(kind="bar", stacked=True)
        .get_figure()
    )
    fig.gca().set_title("Execution time composition")
    fig.gca().set_ylabel("Time [us]")
    fig.gca().set_xticklabels(fig.gca().get_xticklabels(), rotation=0)
    fig.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plot_path = os.path.join(
        plot_dir, "01_image_filter", "execution_time_composition.png"
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)


def reduction():
    # Build binaries
    build_target("04_reduction", "all")
    binaries = [
        os.path.join(git_root, "04_reduction", "build", b)
        for b in [
            "01_seperate_kernels",
            "02_atomic_kernel",
            "03_atomic_kernel_cascade",
            "04_atomic_kernel_opt",
            "05_atomic_kernel_seq_address",
            "06_atomic_kernel_loop_unroll",
            "07_atomic_kernel_shuffle",
        ]
    ]
    kernel_name = "Sum"

    # Profile kernels
    d = []
    for image in images:
        for margin in [1, 2, 3]:
            with tempfile.NamedTemporaryFile() as tmpfile:
                report = NvProfReport(binary, [str(margin), image, tmpfile.name])
                d.append(
                    {
                        "image": os.path.basename(image),
                        "margin": margin,
                        "exec_time": report.get_kernel_execution_time(kernel_name),
                        "launch_time": report.get_kernel_launch_time(),
                        "memcpy_to_device_time": report.get_memcpy_to_device_time(),
                        "memcpy_to_host_time": report.get_memcpy_to_host_time(),
                    },
                )
    df = pd.DataFrame(d)
    df.to_csv(os.path.join(plot_dir, "04_reduction", "kernel_execution_times.csv"))

    # Plot execution times
    fig = (
        df.pivot(index="image", columns="margin", values="exec_time")
        .plot(kind="bar")
        .get_figure()
    )
    fig.gca().set_title("Kernel execution times")
    fig.gca().set_ylabel("Time [us]")
    fig.gca().set_yscale("log", base=2)
    fig.gca().set_xticklabels(fig.gca().get_xticklabels(), rotation=0)
    fig.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plot_path = os.path.join(plot_dir, "01_image_filter", "kernel_execution_times.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)

    # Plot memcpy times
    fig = (
        df[df["margin"] == 1]
        .set_index("image")[
            [
                "launch_time",
                "exec_time",
                "memcpy_to_device_time",
                "memcpy_to_host_time",
            ]
        ]
        .plot(kind="bar", stacked=True)
        .get_figure()
    )
    fig.gca().set_title("Execution time composition")
    fig.gca().set_ylabel("Time [us]")
    fig.gca().set_xticklabels(fig.gca().get_xticklabels(), rotation=0)
    fig.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plot_path = os.path.join(
        plot_dir, "01_image_filter", "execution_time_composition.png"
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)

    # Compare Cascade
    d = []
    binary = binaries[2]
    iter = 1
    for n in range(12, 24):
        problem_size = 2**n
        for c in range(1, 11):
            cascade_size = 2**c
            report = NvProfReport(binary, [str(problem_size), str(cascade_size)])
            d.append(
                {
                    "binary": os.path.basename(binary),
                    "n": problem_size,
                    "c": cascade_size,
                    "exec_time": report.get_kernel_execution_time(kernel_name) * iter,
                    "launch_time": report.get_kernel_launch_time() * iter,
                }
            )
    df = pd.DataFrame(d)
    df.to_csv(
        os.path.join(plot_dir, "04_reduction", "kernel_cascade_execution_times.csv")
    )

    # Plot kernel cascade execution times
    fig = df.pivot(index="n", columns="c", values="exec_time").plot().get_figure()
    fig.gca().set_title("Kernel execution times")
    fig.gca().set_ylabel("Time [us]")
    fig.gca().set_yscale("log", base=2)
    fig.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    fig.gca().set_xlabel("Problem Size")
    fig.gca().set_xscale("log", base=2)
    plot_path = os.path.join(
        plot_dir, "04_reduction", "kernel_cascade_execution_time.png"
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Performance Processor",
        description="Print plots of cuda kernel performance measurements",
    )
    parser.add_argument("exercise", choices=["image_filter", "reduction"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.exercise == "image_filter":
        image_filter()
    elif args.exercise == "reduction":
        reduction()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
