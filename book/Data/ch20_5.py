import matplotlib.pyplot as plt

data1 = [1, 2, 3, 4, 5, 6, 7, 8]        # data1 line
data2 = [1, 4, 9, 16, 25, 36, 49, 64]   # data2 line
data3 = [1, 3, 6, 10, 15, 21, 28, 36]   # data3 line
data4 = [1, 7, 15, 26, 40, 57, 77, 100] # data4 line
data5 = [8, 7, 6, 5, 4, 3, 2, 1]
data6 = [64, 49, 36, 25, 16, 9, 4, 1]
data7 = [36, 28, 21, 15, 10, 6, 3, 1]
data8 = [100, 77, 57, 40, 26, 15, 7, 1]

seq = [1, 2, 3, 4, 5, 6, 7, 8]
plt.plot(seq, data1, 'g--', seq, data2, 'r--', seq, data3, 'y:', seq, data4, 'k. ')
plt.plot(seq, data5, '-*', seq, data6, '-o', seq, data7, '-^', seq, data8, '-s')
plt.title('Test Chart', fontsize=24)
plt.xlabel('x-Value', fontsize=14)
plt.ylabel('y-Value', fontsize=14)
plt.show()