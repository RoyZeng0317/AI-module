import matplotlib.pyplot as plt

data1 = [1, 4, 9, 16, 25, 36, 49, 64]       # data1 line
data2 = [1, 3, 6, 10, 15, 21, 28, 36]       # data2 line
seq = [1, 2, 3, 4, 5, 6, 7, 8]
plt.plot(seq, data1, seq, data2)            # data1 & data2 line
plt.title("Test Chart", fontsize=24)
plt.xlabel("x-Value", fontsize=14)
plt.ylabel("y-Value", fontsize=14)
plt.grid()
plt.show()