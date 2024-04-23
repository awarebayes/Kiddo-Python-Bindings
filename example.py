import kiddo_python_bindings as kpb
import numpy as np

a = np.random.randn(10, 2).astype(np.float32)
tree = kpb.Py2KDTree(a)
zero = np.array([0, 0], dtype=np.float32)
closest = tree.nearest_n_within(zero, 2.0, max_neighbours=10, sorted=True)
within = tree.within(zero, 2.0)
num_close = tree.count_within(zero, 2.0)
print("Closest", closest)
print("Within", within)
print("Num close", num_close)
