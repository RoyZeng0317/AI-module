import matplotlib.pyplot as plt

xpt = list(range(1, 101))
ypt = [x**2 for x in xpt]
plt.scatter(xpt, ypt, color='y')
plt.show()