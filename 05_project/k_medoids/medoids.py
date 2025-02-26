#!/usr/bin/env python3

import sys
import time
from io import StringIO
from sklearn_extra.cluster import KMedoids
import numpy as np

def main(n_clusters, points_path, medoids_path):
    X = np.loadtxt(points_path, delimiter=",")

    time_start = time.perf_counter_ns()
    c = KMedoids(n_clusters=n_clusters).fit(X)
    time_end = time.perf_counter_ns()
    print(f"TRACE: PAM {time_end-time_start} ns", file=sys.stderr)

    with open(medoids_path, "w") as f:
        np.savetxt(f, c.cluster_centers_, fmt="%.18f", delimiter=",")
    #print(f"TRACE: COST {c.inertia_}", file=sys.stderr)

if __name__ == "__main__":
    assert len(sys.argv) == 4, "Usage: python3 medoids.py <n_clusters> <points_path> <medoids_path>"
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])