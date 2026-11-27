# 04｜RNN 循環神經網路

[← 目錄](README.md) ｜ [下一章：LSTM →](05_LSTM.md)

## 核心概念

**RNN（Recurrent Neural Network）依序處理資料，並把前一步的隱藏狀態傳到下一步。**

```text
「我」 → 「今天」 → 「打」 → 「羽球」
 h1       h2       h3        h4
  └──────→└──────→└────────→┘
```

常見任務：文字序列、時間序列、感測器數據。一般 RNN 可能面臨**長期依賴**與**梯度消失／爆炸**問題。

## 簡化數學概念

$$h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$$

當前狀態 $h_t$ 同時受到目前輸入 $x_t$ 和上一狀態 $h_{t-1}$ 影響。

## Python：PyTorch RNN

```python
import torch
from torch import nn

torch.manual_seed(42)
rnn = nn.RNN(input_size=3, hidden_size=5, batch_first=True)
# 2 組序列，每組 4 個時間點，每點 3 個特徵
x = torch.randn(2, 4, 3)
output, last_hidden = rnn(x)
print('各時間點輸出:', output.shape)      # [2, 4, 5]
print('最後隱藏狀態:', last_hidden.shape)  # [1, 2, 5]
```

## RNN vs CNN

- CNN：學習圖片的**空間結構**。
- RNN：學習資料的**時間／順序關係**。
- CNN + RNN：可用於逐幀提取影像特徵後分析影片序列。

## 延伸

[LSTM：改善長期記憶](05_LSTM.md) ｜ [Transformer：Attention](06_Transformer.md)
