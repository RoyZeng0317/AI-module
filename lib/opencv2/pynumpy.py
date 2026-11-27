import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(type(a))
print(a[0], a[1], a[2], a[3], a[4])
a[0] = 5
print(a)
a[3] = 0
print(a)
b = np.array([[1, 7, 2], [3, 6, 8]])
print(b[0, 0], b[0, 1], b[0, 2])
print(b[1, 0], b[1, 1], b[1, 2])
b[0, 0] = 6
b[1, 2] = 1
print(b)
c = np.array([[1, 7, 4], [5, 2, 8]])
print(c)
d = np.array([1, 2, 3, 4, 5], int)
print(d)
e = np.array((1, 2, 3, 4, 5), dtype=float)
print(e)
f = np.array((3.14, 6.28, 5.73, 1.6928, 1.732), dtype=float)
print(f)
print(f"矩陣總數: {f.size}, NxM陣列形狀: {f.shape}, 陣列元素占用的位元組數: {f.itemsize}, {f.ndim}維陣列, 整個陣列占用的位元組數: {f.nbytes}")
g = np.array([1, 2, 3, 4, 5, 6])
print(g)
h = g.reshape((3, 2))
print(h, f"{h.ndim}維矩陣, 矩陣總數: {h.size}, NxM矩陣形狀: {f.shape}")