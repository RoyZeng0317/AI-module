import matplotlib.pyplot as plt

x = [x for x in range(10)]       # generate 0, 1, ... 9 of string
squars = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
plt.plot(x, squars)
plt.show()