#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from cost import cost

def main(points_path, medoids_path, output_path):
    points = np.loadtxt(points_path, delimiter=",")
    medoids = np.loadtxt(medoids_path, delimiter=",")

    labels = np.array([np.argmin([np.linalg.norm(p-m) for m in medoids]) for p in points])
    total_cost = cost(points_path, medoids_path)

    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', marker='o', label='Points')
    plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='x', s=100, label='Medoids')


    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot([], [], ' ', label="cost = {cost:.0f}".format(cost=total_cost))
    plt.title('K-Medoids Clustering')
    plt.legend()
    plt.savefig(output_path)

if __name__ == "__main__":
    assert len(sys.argv) == 4, "Usage: python3 visualize.py <points_path> <medoids_path> <output_path>"
    main(sys.argv[1], sys.argv[2], sys.argv[3])