#!/usr/bin/env python3

from sklearn.datasets import make_blobs
from io import StringIO
import numpy as np

def datagen(n_samples, centers, f):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.5, random_state=0)
    np.savetxt(f, X, fmt="%.18f", delimiter=",")


def main():
    f = StringIO()
    datagen(10**4, 5, f)
    print(f.getvalue().strip())

if __name__ == "__main__":
    main()