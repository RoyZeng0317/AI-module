import numpy as np
from pathlib import Path
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

"""
1. 模擬非線性可分的資料
make_moons 產生兩個交錯的半月形分佈，線性邊界切不開，
用來凸顯核方法(Kernel Method)存在的意義：
靠核函數(kernel function)把資料隱式投影到高維空間，
讓原本線性不可分的資料變成可分。
"""
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'kernel_svm_model.pkl'

x, y = make_moons(n_samples=300, noise=0.25, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

"""
2. 對照組：線性核(linear kernel)
沒有核技巧(kernel trick)、只能切出一條直線邊界，
在半月形資料上表現會明顯較差。
"""
linear_model = SVC(kernel='linear')
linear_model.fit(x_train, y_train)
linear_pred = linear_model.predict(x_test)
linear_acc = accuracy_score(y_test, linear_pred)
print(f"[linear kernel] Accuracy: {linear_acc:.4f}")

"""
3. 核方法：RBF 核(Radial Basis Function kernel)
gamma 控制核函數的影響範圍，C 控制誤差容忍度(正則化強度)。
RBF 核能畫出非線性邊界，理論上應該比 linear kernel 更貼合半月形資料。
"""
rbf_model = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_model.fit(x_train, y_train)
rbf_pred = rbf_model.predict(x_test)
rbf_acc = accuracy_score(y_test, rbf_pred)
print(f"[rbf kernel] Accuracy: {rbf_acc:.4f}")

"""
4. 儲存 RBF 核模型
"""
joblib.dump(rbf_model, MODEL_PATH)

loaded_model = joblib.load(MODEL_PATH)
sample = np.array([[0.5, 0.5]])
result = loaded_model.predict(sample)
print(f"Loaded model prediction for {sample.tolist()}: {result.tolist()}")
