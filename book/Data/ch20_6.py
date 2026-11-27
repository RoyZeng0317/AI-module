import matplotlib.pyplot as plt

Benz = [3367, 4120, 5539]           # Benz line
BMW = [4000, 3590, 4423]            # BMW line
Lexus = [5200, 4930, 5350]          # Lexus line
seq = [2021, 2022, 2023]            # year

plt.plot(seq, Benz, '-*', seq, BMW, '-o', seq, Lexus, '-^')
plt.title("Sakes Report", fontsize=24)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Sales", fontsize=14)
plt.xticks(seq)
plt.show()