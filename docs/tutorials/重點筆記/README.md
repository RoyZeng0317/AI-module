# AI 與機器學習學習筆記 📚

以**繁體中文**整理，從基礎到影像辨識、序列模型、Transformer 與 RAG。每章包含原理、示範程式碼與複習重點。

## 建議學習路線

1. [機器學習全貌與訓練流程](00_學習地圖與基礎.md)
2. [決策樹 Decision Tree](01_決策樹.md)
3. [梯度下降 Gradient Descent](02_梯度下降.md)
4. [CNN 卷積神經網路](03_CNN.md)
5. [RNN 循環神經網路](04_RNN.md)
6. [LSTM 長短期記憶](05_LSTM.md)
7. [Transformer 與 Attention](06_Transformer.md)
8. [RAG 檢索增強生成](07_RAG.md)
9. [OCR 光學字元辨識](08_OCR.md)
10. [綜合比較與複習題](09_總整理與測驗.md)

## 使用方法

- VS Code：開啟資料夾，按 `Ctrl+Shift+V` 預覽 Markdown。
- Obsidian：將此資料夾作為筆記庫或複製到現有筆記庫。
- 程式範例以 Python 3.10+ 撰寫；依章節安裝套件。
- **範例是教學用途，不代表真實模型的準確率或生產環境設定。**

## 程式環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scikit-learn matplotlib torch
# OCR 章節另需：pip install pillow pytesseract
# RAG 章節使用 Python 標準函式庫，不必安裝額外套件
```

> `pytesseract` 還需要額外安裝 Tesseract OCR 執行程式；詳見 OCR 章節。

## 主題關聯

```text
資料 → 模型 → 預測 → Loss → Gradient Descent → 更新參數
       ├─ 決策樹：條件分裂（一般不使用梯度下降）
       ├─ CNN：圖片的空間特徵
       ├─ RNN / LSTM：序列資訊
       └─ Transformer：Attention 建模關聯

圖片文字 → OCR → 文字
問題 → 檢索資料（RAG）→ LLM 產生有依據的答案
```
