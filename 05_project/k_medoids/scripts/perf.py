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

def main(report_path):
    # Build binaries
    build_target("05_project/k_medoids", "all")
    binaries = [
        os.path.join(git_root, "05_project", "k_medoids", b)
        #TODO: use all kernels
        for b in [
            "medoids.py",
            #"build/01_pam",
            #"build/02_pam_no_sqrt",
            #"build/03_pam_2d",
            #"build/04_pam_single_precision",
            #"build/05_pam_intrinsics",
            #"build/06_pam_no_pow",
            #"build/07_pam_shmem",
            #"build/08_pam_small_block",
            "build/09_pam_end_early",
            #"build/10_pam_next",
            #"build/11_pam_matrix",
        ]
    ]
    #TODO: increase
    n_clusters = [3]
    #TODO: increase
    n_points = [10**2, 10**3]
    #TODO: increase
    iterations = range(1)
    points_path = os.path.join(git_root, "05_project", "k_medoids", "build", "data.csv")
    kernel_name = "swap_cost"

    # Profile kernels
    d = []
    for binary, n_point, n_cluster, _ in itertools.product(binaries, n_points, n_clusters, iterations):
        with tempfile.NamedTemporaryFile() as points, tempfile.NamedTemporaryFile() as medoids:
            datagen(n_point, n_cluster, points)
            report = NvProfReport(binary, [str(n_cluster), points.name, medoids.name])
            d.append(
                {
                    "binary": os.path.basename(binary),
                    "n_clusters": n_cluster,
                    "n_points": n_point,
                    "total_time": report.get_traced_value("PAM") / 1000,
                    "total_kernel_exec_time": report.get_kernel_exec_stats(kernel_name)["time"],
                    "total_kernel_launch_time": report.get_kernel_launch_stats()["time"],
                    "avg_kernel_exec_time": report.get_kernel_exec_stats(kernel_name)["avg"],
                    "avg_kernel_launch_time": report.get_kernel_launch_stats()["avg"],
                    "iterations": report.get_kernel_exec_stats(kernel_name)["instances"],
                    "memcpy_host_to_device_time": report.get_memcpy_to_device_stats()["time"],
                    "memcpy_device_to_host_time": report.get_memcpy_to_device_stats()["time"],
                    "cost": cost(points_path, medoids.name),
                }
        )
    df = pd.DataFrame(d)
    df = df.groupby(['binary', 'n_clusters', 'n_points']).mean().reset_index()
    df.to_csv(report_path, index=False)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 perf.py <report_path>"
    main(sys.argv[1])
