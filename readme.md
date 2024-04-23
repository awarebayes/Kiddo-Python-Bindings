# Kiddo Python Bindings

Barebones python bindings for the [kiddo](https://github.com/sdd/kiddo) library.

You will probably want to change and recompile if you want to use it.

As of now, only f32 2D trees are supported, because that's what I need. You can change it to 3D/4D/AnyD.

Usage

```sh
git clone https://github.com/awarebayes/Kiddo-Python-Bindings
maturin develop --release
```

Or simply

```
pip install kiddo-python-bindings
```

> Make sure your data is in float32 before passing it to functions

```python

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
```

```python
Closest (
    array([0, 9, 5, 7, 6, 4], dtype=uint64),
    array([0.08116178, 0.25775522, 0.3743909 , 1.0922432 , 1.3634278 ,
       1.9392966 ], dtype=float32)
)
Within (
    array([0, 4, 5, 6, 7, 9], dtype=uint64),
    array([0.08116178, 1.9392966 , 0.3743909 , 1.3634278 , 1.0922432 ,
       0.25775522], dtype=float32)
)
Num close 6
```
