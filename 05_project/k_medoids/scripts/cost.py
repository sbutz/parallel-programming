#!/usr/bin/env python3

import numpy as np
import sys

def cost(points_path, medoids_path):
    points = np.loadtxt(points_path, delimiter=",")
    medoids = np.loadtxt(medoids_path, delimiter=",")

    return np.sum([min([np.linalg.norm(p-m) for m in medoids]) for p in points])

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python3 cost.py <points_path> <medoids_path>"
    print(cost(sys.argv[1], sys.argv[2]))