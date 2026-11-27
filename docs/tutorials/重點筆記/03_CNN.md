# 03｜CNN 卷積神經網路

[← 目錄](README.md) ｜ [下一章：RNN →](04_RNN.md)

## 核心概念

**CNN（Convolutional Neural Network）擅長學習影像的局部空間特徵。**

```text
圖片 → 卷積 Convolution → ReLU → 池化 Pooling
     → 更多特徵層 → 分類器 → 貓／狗
```

- **卷積層**：以可學習的濾波器偵測邊緣、紋理等特徵。
- **ReLU**：`max(0, x)`，提供非線性。
- **Pooling**：降低特徵圖空間尺寸，常用 Max Pooling。
- **Flatten / Global Average Pooling**：將特徵整理後接分類器。

## Max Pooling 手算

對以下 $4\times4$ 特徵圖，使用 `kernel_size=2, stride=2`：

```text
1 3 | 2 4
5 6 | 1 2
----+----
7 2 | 8 3
1 4 | 2 9
```

每個 $2\times2$ 區塊取最大值，輸出：

```text
6 4
7 9
```

## Python：純 Python Max Pooling

```python
feature = [
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [7, 2, 8, 3],
    [1, 4, 2, 9],
]
pooled = []
for i in range(0, 4, 2):
    row = []
    for j in range(0, 4, 2):
        block = [feature[i + di][j + dj] for di in range(2) for dj in range(2)]
        row.append(max(block))
    pooled.append(row)
print(pooled)  # [[6, 4], [7, 9]]
```

## Python：PyTorch CNN 前向傳播

安裝：`pip install torch`

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(8 * 16 * 16, 2),  # 輸入為 32x32，池化後 16x16
)

images = torch.randn(4, 3, 32, 32)  # 4 張 RGB 假圖片
logits = model(images)
labels = torch.tensor([0, 1, 0, 1])
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
print('輸出形狀:', logits.shape)  # [4, 2]
print('Loss:', loss.item())
```

> 這裡只示範一次前向／反向傳播，尚未呼叫 optimizer 更新參數；不是訓練完成的辨識模型。`CrossEntropyLoss` 直接接 logits，**不要先手動 Softmax**。

## 注意

Pooling 會減少解析度，也可能遺失細節；現代 CNN 並非都使用 Max Pooling，有些以 stride convolution 取代。

## 延伸

[梯度下降](02_梯度下降.md) ｜ [OCR：辨識影像文字](08_OCR.md)
