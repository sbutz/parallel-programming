#!/usr/bin/env python3

from cost import cost
from datagen import datagen
from nv_prof import NvProfReport
import itertools
import logging
import os
import pandas as pd
import subprocess
import sys
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

def profile(binary, n_points, n_clusters, kernel_name):
    logger.info(f"Profiling ${binary}. n_points={n_points}, n_clusters={n_clusters}")
    with tempfile.NamedTemporaryFile() as points, tempfile.NamedTemporaryFile() as medoids:
        datagen(n_points, n_clusters, points)
        report = NvProfReport(binary, [str(n_clusters), points.name, medoids.name])
        return {
            "binary": os.path.basename(binary),
            "n_clusters": n_clusters,
            "n_points": n_points,
            "total_time": report.get_traced_value("PAM") / 1000,
            "total_kernel_exec_time": report.get_kernel_exec_stats(kernel_name)["time"],
            "total_kernel_launch_time": report.get_kernel_launch_stats()["time"],
            "avg_kernel_exec_time": report.get_kernel_exec_stats(kernel_name)["avg"],
            "avg_kernel_launch_time": report.get_kernel_launch_stats()["avg"],
            "iterations": report.get_kernel_exec_stats(kernel_name)["instances"],
            "memcpy_host_to_device_time": report.get_memcpy_to_device_stats()["time"],
            "memcpy_device_to_host_time": report.get_memcpy_to_device_stats()["time"],
            "cost": cost(points.name, medoids.name),
        }

def main(report_path):
    # Build binaries
    build_target("05_project/k_medoids", "all")

    d = []
    iterations = range(10)
    kernel_name = "swap_cost"

    binaries = [
        os.path.join(git_root, "05_project", "k_medoids", b)
        for b in [
            "build/01_pam",
            "build/02_pam_no_sqrt",
            "build/03_pam_2d",
            "build/04_pam_single_precision",
            "build/05_pam_intrinsics",
            "build/06_pam_no_pow",
            "build/07_pam_shmem",
            "build/08_pam_small_block",
        ]
    ]
    n_clusters = [3, 5, 10]
    n_points = [10**2, 10**3, 10**4]
    for binary, n_point, n_cluster, _ in itertools.product(binaries, n_points, n_clusters, iterations):
        # Distance Matrix used in python implementaiton is limited by memory
        if binary.endswith(".py") and n_point > 3*10**4:
            continue
        d.append(profile(binary, n_point, n_cluster, kernel_name))

    binaries = [
        os.path.join(git_root, "05_project", "k_medoids", b)
        for b in [
            "medoids.py",
            "build/09_pam_end_early",
        ]
    ]
    n_clusters = [3, 5, 10]
    n_points = [10**2, 10**3, 10**4, 10**5]
    for binary, n_point, n_cluster, _ in itertools.product(binaries, n_points, n_clusters, iterations):
        # Distance Matrix used in python implementaiton is limited by memory
        if binary.endswith(".py") and n_point > 3*10**4:
            continue
        d.append(profile(binary, n_point, n_cluster, kernel_name))

    # Aggregate Results
    df = pd.DataFrame(d)
    df = df.groupby(['binary', 'n_clusters', 'n_points']).mean().reset_index()
    df.to_csv(report_path, index=False)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 perf.py <report_path>"
    main(sys.argv[1])
