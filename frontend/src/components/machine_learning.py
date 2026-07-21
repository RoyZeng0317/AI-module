# 匯入 pandas 套件
import pandas as pd
# 匯入 joblib 套件
import joblib
from pathlib import Path

# 用腳本自己的路徑定位檔案，不管從哪個資料夾執行這支程式都能找到 data.csv
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data.csv'
MODEL_PATH = BASE_DIR / 'linear_regression_model.pkl'

data = pd.read_csv(DATA_PATH)

print(data)


# 切分資料
from sklearn.model_selection import train_test_split

x = data[["height"]]
y = data[["weight"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 建立模型(線性回歸)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
# 開始學習
model.fit(x_train, y_train)

# 模型測試
predictions = model.predict(x_test)
print(predictions)

# 評估模型
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# 儲存模型

joblib.dump(model, MODEL_PATH)

# 讀取模型
model = joblib.load(MODEL_PATH)

# 使用模型進行預測
# 用 DataFrame（欄名對齊 "height"）而非原始 list，避免 sklearn 跳出
# "X does not have valid feature names" 警告
new_height = pd.DataFrame([[170]], columns=["height"])
result = model.predict(new_height)
print(result)
