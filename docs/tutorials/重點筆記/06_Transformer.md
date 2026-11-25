# 06｜Transformer 與 Attention

[← 目錄](README.md) ｜ [下一章：RAG →](07_RAG.md)

## 核心概念

**Transformer 主要使用 Attention（注意力機制）來建立序列中不同位置的關聯，而不依賴 RNN 那種逐步傳遞的循環結構。**

例如：「小明把球拍交給阿華，因為**他**要上場。」要判斷「他」指誰，需要看整句的上下文；注意力機制能讓不同 token 互相參考。

## 常見術語

- **Token**：文字被切分後的處理單位。
- **Embedding**：將 token 轉為向量表示。
- **Positional Encoding / Position Information**：提供位置或順序資訊。
- **Self-Attention**：序列內部位置彼此參考。
- **Multi-Head Attention**：使用多組注意力頭學習不同關係。
- **Feed-Forward Network（FFN）**：對每個位置做非線性轉換。

## Attention 的簡化公式

$$Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q（Query）**：目前在找什麼資訊。
- **K（Key）**：每個位置提供什麼匹配線索。
- **V（Value）**：要彙整的實際資訊。

## Python：手寫縮放點積注意力

安裝：`pip install torch`

```python
import torch
import math

torch.manual_seed(42)
Q = torch.randn(3, 4)  # 3 個 token，每個 4 維
K = torch.randn(3, 4)
V = torch.randn(3, 4)

scores = Q @ K.T / math.sqrt(K.shape[-1])
weights = torch.softmax(scores, dim=-1)
result = weights @ V
print('注意力權重:', weights)
print('輸出形狀:', result.shape)  # [3, 4]
```

> 此範例示範**單頭、無遮罩**注意力的核心運算，不是完整 Transformer。實際自回歸語言模型通常需要 causal mask，避免偷看未來 token。

## Python：使用 PyTorch 多頭注意力

```python
import torch
from torch import nn

attention = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
x = torch.randn(2, 4, 8)
y, weights = attention(x, x, x)
print(y.shape)        # [2, 4, 8]
print(weights.shape)  # [2, 4, 4]（預設平均各 head 權重）
```

## Transformer 與 RNN／LSTM 的差別

- RNN／LSTM：依序更新狀態。
- Transformer：藉 Attention 讓序列位置直接建立關聯，訓練時通常更易平行化。
- 注意力計算對長序列可能耗用較多記憶體；具體成本依實作而異。

## 延伸

[LSTM](05_LSTM.md) ｜ [RAG：為 LLM 補充外部知識](07_RAG.md) ｜ [梯度下降](02_梯度下降.md)
