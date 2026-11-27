import matplotlib.pyplot as plt

squares = [0, 1, 4, 9, 16, 25, 36, 49, 64]
plt.plot(squares)                   # string squares data is y  of value
plt.axis((0, 8, 0, 70))             # x  0-8, y  0-70
plt.grid()
plt.show()