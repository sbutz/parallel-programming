# parallel-programming
Shared Workspace for parallel programming workshop

## Setup
Stefan's Setup using a GCP (Google Cloud) Compute Engine instance is described in [here](./Setup.md).

## Profling
For guides on profiling see [here](./Profiling.md).

There is a script to automatically run performance measurements for the different implementations.
It will generate plots to visualize the results.
```bash
./plots/perf.py image_filter  # Exercise 1
./plots/perf.py reduction     # Exercise 4
```