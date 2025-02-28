#!/usr/bin/env python3

import os
import pandas as pd
import subprocess
import sys
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

git_root = (
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, check=True
    )
    .stdout.decode("utf-8")
    .strip()
)
plot_dir = os.path.join(git_root, "05_project", "k_medoids", "plots")


def main(report_path):
    df = pd.read_csv(report_path)

    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None
    # pd.options.display.float_format = '{:.2f}'.format
    # print(df)

    # Plot first kernels
    filtered_df = df[
        (df["n_clusters"] == 3)
        & (df["binary"].isin(["medoids.py", "01_pam", "02_pam_no_sqrt"]))
    ]
    filtered_df["total_time"] = filtered_df["total_time"] / 1000 / 1000
    fig = (
        filtered_df.pivot(index="n_points", columns="binary", values="total_time")
        .plot(kind="line", label="n_clusters = 3")
        .get_figure()
    )
    ax = fig.gca()
    ax.set_title("Execution Time")
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Total Time [s]")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.legend(title="n_clusters = 3")  # Add legend with title
    fig.savefig(os.path.join(plot_dir, "01_perf_kernel_01_02.png"))

    # Single Precision Loss in Cost - Total Time
    filtered_df = df[
        (df["n_clusters"] == 3)
        & (df["n_points"] == 10**4)
        & (
            df["binary"].isin(
                [
                    "medoids.py",
                    "03_pam_2d",
                    "04_pam_single_precision",
                    "05_pam_intrinsics",
                    "06_pam_no_pow",
                ]
            )
        )
    ]
    filtered_df["total_time"] = filtered_df["total_time"] / 1000

    # Plot total time
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
    colors = plt.cm.tab20.colors  # Use a colormap for different colors
    filtered_df.plot(
        kind="bar", x="binary", y="total_time", ax=ax, legend=False, color=colors
    )
    ax.set_title("Execution Time")
    ax.set_xlabel("Binary")
    ax.set_ylabel("Total Time [ms]")
    ax.set_yscale("log", base=2)  # Set y-axis to logarithmic scale with base 2
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
    fig.tight_layout()  # Adjust layout to fit labels
    fig.savefig(os.path.join(plot_dir, "02_perf_sp_loss_total_time.png"))

    # Plot cost
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
    filtered_df.plot(
        kind="bar", x="binary", y="cost", ax=ax, legend=False, color=colors
    )
    ax.set_title("Cost")
    ax.set_xlabel("Binary")
    ax.set_ylabel("Cost")
    ax.set_yscale("log", base=2)  # Set y-axis to logarithmic scale with base 2
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
    fig.tight_layout()  # Adjust layout to fit labels
    fig.savefig(os.path.join(plot_dir, "02_perf_sp_loss_cost.png"))

    # Compare Runtime of 06 and 07
    filtered_df = df[
        (df["n_clusters"] == 3)
        & (df["binary"].isin(["06_pam_no_pow", "07_pam_shmem", "08_pam_small_block"]))
    ]
    filtered_df["total_time"] = filtered_df["total_time"] / 1000  # Convert to seconds
    fig = (
        filtered_df.pivot(index="n_points", columns="binary", values="total_time")
        .plot(kind="line")
        .get_figure()
    )
    ax = fig.gca()
    ax.set_title("Compare runtime of kernels with shared memory and smaller block size")
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Total Time [ms]")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.legend(title="n_clusters = 3")  # Add legend with title
    fig.tight_layout()  # Adjust layout to fit labels
    fig.savefig(os.path.join(plot_dir, "03_runtime_comparison_06_07.png"))

    # Plot all binaries with n_clusters = 3 and n_points up to 10,000
    filtered_df = df[(df["n_clusters"] == 3) & (df["n_points"] <= 10000)]
    filtered_df["total_time"] = (
        filtered_df["total_time"] / 1000
    )  # Convert to milliseconds

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
    filtered_df.pivot(index="n_points", columns="binary", values="total_time").plot(
        kind="line", ax=ax
    )
    ax.set_title("Total Runtime for All Binaries")
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Total Time [ms]")
    ax.set_yscale("log", base=2)  # Set y-axis to logarithmic scale with base 2
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.legend(title="Binaries")
    fig.tight_layout()  # Adjust layout to fit labels
    fig.savefig(os.path.join(plot_dir, "04_total_runtime_all_binaries.png"))

    # Plot composition of total runtime for the last binary '09_pam_end_early' with n_points=10000
    filtered_df = df[
        (df["binary"] == "09_pam_end_early") & (df["n_points"] == 10000)
    ].iloc[-1]

    total_time = filtered_df["total_time"]
    total_kernel_exec_time = filtered_df["total_kernel_exec_time"]
    total_kernel_launch_time = filtered_df["total_kernel_launch_time"]
    memcpy_time = (
        filtered_df["memcpy_host_to_device_time"]
        + filtered_df["memcpy_device_to_host_time"]
    )
    cpu_time = total_time - (
        total_kernel_exec_time + total_kernel_launch_time + memcpy_time
    )

    labels = ["Kernel Execution Time", "Kernel Launch Time", "Memcpy Time", "CPU Time"]
    sizes = [total_kernel_exec_time, total_kernel_launch_time, memcpy_time, cpu_time]
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.85,
        labeldistance=1.1,
    )
    # Create legend with percentage values
    legend_labels = [
        f"{label}: {size / total_time * 100:.1f}%" for label, size in zip(labels, sizes)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Components",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Composition of Total Runtime")
    fig.tight_layout()  # Adjust layout to fit labels
    fig.savefig(os.path.join(plot_dir, "05_composition_total_runtime.png"))

    # Plot total time for '09_pam_end_early' with different n_clusters
    filtered_df = df[df["binary"] == "09_pam_end_early"]
    filtered_df["total_time"] = (
        filtered_df["total_time"] / 1000
    )  # Convert to milliseconds

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
    filtered_df.pivot(index="n_points", columns="n_clusters", values="total_time").plot(
        kind="line", ax=ax
    )
    ax.set_title("Runtime for different problem sizes")
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Total Time [ms]")
    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.legend(title="n_clusters")
    fig.tight_layout()  # Adjust layout to fit labels
    fig.savefig(os.path.join(plot_dir, "06_total_time_for_different_problem_sizes.png"))


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 plots.py <report_path>"
    main(sys.argv[1])
