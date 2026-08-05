# RNN.py 說明

對應檔案：[RNN.py](RNN.py)

## 這個檔案在做什麼

定義了一個 `SeqLSTM`（LSTM，屬於 RNN 家族的一種）序列分類模型，見 [RNN.py:6-28](RNN.py#L6-L28)。

LSTM 層讀入一段序列，配合隱藏狀態 `h0`／記憶胞狀態 `c0`（RNN 特有、CNN 沒有的東西，用來記住「前面看過什麼」），只取序列最後一個時間點的輸出，丟進 `Linear` 分類。

## 不是在訓練圖片

input shape 是 `(Batch_Size, Sequence_Length, Input_Size)`，也就是「一批、每批有幾個時間步、每個時間步幾個特徵」，跟 [CNN.py](CNN.py) 的 `(Batch, Channel, Height, Width)` 圖片張量完全不同結構。

RNN/LSTM 天生是為了處理**有前後順序關係的序列資料**，例如文字、語音、時間序列——這也符合專案裡其他用到 GRU 的檔案（[chats.py](chats.py)、[voice_clone.py](voice_clone.py)、[character_model.py](character_model.py)）都是拿來處理文字／語音，不是圖片。

`RNN.py` 比較像是獨立的 LSTM 架構範本／教學示範，**不屬於圖片訓練的任何部分**。

## 跟 CNN.py 的關鍵差異：這裡沒有梯度下降

[RNN.py:30-46](RNN.py#L30-L46) 只是「初始化 + 跑一次 forward 檢查輸出形狀對不對」，完全沒有：

- 沒有 loss function（沒有 criterion）
- 沒有 optimizer
- 沒有 `loss.backward()` / `optimizer.step()`

所以這支檔案**目前只是架構驗證用的 smoke test**，不是像 CNN.py 那樣的完整訓練腳本。

## 如果之後要真的訓練 LSTM

需要仿照 CNN.py 第 3、4 段補上：

1. `DataLoader`（把資料包成可迭代的 batch）
2. `criterion`（損失函數，例如 `CrossEntropyLoss`）
3. `optimizer`（例如 `Adam`）
4. 訓練迴圈裡呼叫 `loss.backward()` 與 `optimizer.step()`，才會有真正的梯度下降、模型才會學習。
