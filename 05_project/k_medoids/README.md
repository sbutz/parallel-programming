# K-Medoids

The k-medoids problem is a clustering problem. [[1]](https://doi.org/10.1002/9780470316801.ch2)
The goal is to partion a dataset of `n` points into `k` clusters by minimizing the distance of points in a cluster
and a designated cluster center (medoid).
The medoid has to be an actual datapoint containted in the dataset as compared to k-means.
This allows for greater interpretability of the cluster centers.
Different dissimilarity measures can be used, whereas k-means usally requires euclididan distances.
Since it is a np-hard problem, heursistic solutions exist.

## Run
```
make run
```
