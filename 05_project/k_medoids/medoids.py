#!/usr/bin/env python3

import sys
from io import StringIO
from sklearn_extra.cluster import KMedoids
import numpy as np

def main(n_clusters, path):
    X = np.loadtxt(path, delimiter=",")
    c = KMedoids(n_clusters=n_clusters).fit(X)

    f = StringIO()
    np.savetxt(f, c.cluster_centers_, fmt="%.18f", delimiter=",")
    print(f.getvalue().strip())
    #print(c.inertia_)

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python3 medoids.py <n_clusters> <path>"
    main(int(sys.argv[1]), sys.argv[2])