import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("wine.csv", header=None)
x, y = df.iloc[:, 1:], df.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=5
)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

accuracy = round(model.score(x_test, y_test) * 100, 2)
print(f"精確度: {accuracy:.2f}%")

samples = [
    [1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172],
    [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1, 3.92, 1065],
    [13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720],
]

for i, result in enumerate(model.predict(samples), 2):
    print(f"第{i}小題: 第{result}類")
