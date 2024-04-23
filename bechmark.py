import numpy as np
from scipy.spatial import KDTree
import kiddo_python_bindings as kpb
import time
from random import random
import matplotlib.pyplot as plt


sizes = [10, 50, 100, 300, 500, 750, 1_000, 5_000, 10_000, 50_000, 100_000]
nearest_n_kiddo = []
nearest_n_scipy = []
time_bench = 100

for size in sizes:
    K = min(512, int(size * 0.1))
    points = np.random.randn(size, 2).astype(np.float32)
    tree = kpb.Py2KDTree(points)

    times = []

    for i in range(time_bench):
        rand_point = np.random.randn(2).astype(np.float32)
        rand_radius = abs(random() * 3)
        start = time.time()
        tree.nearest_n_within(rand_point, rand_radius, K, True)
        end = time.time()
        times.append(end - start)

    median_time = np.median(times)
    nearest_n_kiddo.append(median_time)


for size in sizes:
    K = min(512, int(size * 0.1))
    points = np.random.randn(size, 2).astype(np.float32)
    tree = KDTree(points)

    times = []

    for i in range(time_bench):
        rand_point = np.random.randn(2).astype(np.float32)
        rand_radius = abs(random() * 3)
        start = time.time()
        tree.query(rand_point, k=K, distance_upper_bound=rand_radius)  # does not sort
        end = time.time()
        times.append(end - start)

    median_time = np.median(times)
    nearest_n_scipy.append(median_time)

fig, ax = plt.subplots()

ax.plot(sizes, nearest_n_kiddo, label="Kiddo nearest")
ax.plot(sizes, nearest_n_scipy, label="SciPy nearest")
ax.set_xscale("log")
ax.set_xticks(sizes)
ax.set_yscale("log")
ax.grid()
ax.legend()
plt.xlabel("n points")
plt.ylabel("seconds")
plt.title("nearest within radius min(k=0.1 * size, 512)")
plt.show()
