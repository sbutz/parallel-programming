#!/usr/bin/env python3

import sys
from io import StringIO
from sklearn_extra.cluster import KMedoids
import numpy as np

def main(path):
    X = np.loadtxt(path, delimiter=",")
    c = KMedoids(n_clusters=3).fit(X)

    f = StringIO()
    np.savetxt(f, c.cluster_centers_, fmt="%.18f", delimiter=",")
    print(f.getvalue().strip())

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 medoids.py <path>"
    main(sys.argv[1])