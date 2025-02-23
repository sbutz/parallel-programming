#!/usr/bin/env python3

from sklearn.datasets import make_blobs
from io import StringIO
import numpy as np

def main():
    X, y = make_blobs(n_samples=3000, centers=5, cluster_std=0.5, random_state=0)

    f = StringIO()
    np.savetxt(f, X, fmt="%.18f", delimiter=",")
    print(f.getvalue().strip())

if __name__ == "__main__":
    main()