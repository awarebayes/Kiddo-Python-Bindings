import kiddo_python_bindings as kpb
import numpy as np

a = np.random.randn(10, 2).astype(np.float32)
tree = kpb.Py2KDTree(a)
closest = tree.nearest_n_within(np.array([0, 0], dtype=np.float32), 2.0, 10, True)
num_close = tree.count_within(np.array([0, 0], dtype=np.float32), 2.0)
print("Closest", closest)
print("Num close", num_close)
