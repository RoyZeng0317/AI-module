# 05｜LSTM 長短期記憶網路

[← 目錄](README.md) ｜ [下一章：Transformer →](06_Transformer.md)

## 一句話理解

**LSTM（Long Short-Term Memory）是 RNN 的一種改良架構，透過記憶狀態和閘門，更容易保留較長距離的重要資訊。**

例如：「小明打了兩小時羽球，之後洗澡、吃飯、寫作業，晚上覺得很＿＿。」模型需要保留前面「運動兩小時」的資訊。

## 三個閘門

| 閘門 | 英文 | 作用 |
|---|---|---|
| 遺忘閘門 | Forget Gate | 控制保留多少舊記憶 |
| 輸入閘門 | Input Gate | 控制寫入多少新資訊 |
| 輸出閘門 | Output Gate | 控制輸出哪些狀態資訊 |

> 閘門的值是透過訓練學到的連續數值，不是人工寫死的「記住／忘掉」二選一。

## Python：PyTorch LSTM

```python
import torch
from torch import nn

torch.manual_seed(42)
lstm = nn.LSTM(input_size=3, hidden_size=6, batch_first=True)
x = torch.randn(2, 5, 3)  # batch=2, seq_len=5, features=3
output, (h_n, c_n) = lstm(x)
print('output:', output.shape)  # [2, 5, 6]
print('h_n:', h_n.shape)        # [1, 2, 6]
print('c_n:', c_n.shape)        # [1, 2, 6]
```

其中 `h_n` 是最終隱藏狀態，`c_n` 是最終 cell state（記憶狀態）。

## LSTM 的限制

- 通常依時間步循序運算，不如 Transformer 容易平行化序列位置。
- 長序列仍可能有計算成本與資訊保留限制。
- 並不是所有序列問題都比 RNN 或 Transformer 好。

## 延伸

[RNN](04_RNN.md) ｜ [Transformer](06_Transformer.md)
