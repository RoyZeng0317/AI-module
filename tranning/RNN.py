import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""
1. 定義 LSTM 模型架構
"""
class SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 層
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 最終輸出分類層
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch_Size, Sequence_Length, Input_Size)

        # init 隱藏狀態(h0) and 記憶胞狀態(c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向傳播 LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 只取最後一個時間步 -> (Batch, num_classes)
        return out

"""
2. 基本參數與模擬序列數據
假設輸入特徵數 = 10, 序列長度 = 20, LSTM 隱藏維度 = 64, 2 層 LSTM，最後分類 2 類
模擬 100 筆序列與對應標籤
"""
input_size, sql_length, hidden_size, num_layers, num_classes = 10, 20, 64, 2, 2

dummy_x = torch.randn(100, sql_length, input_size)
dummy_y = torch.randint(0, num_classes, (100,))

dataset = TensorDataset(dummy_x, dummy_y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

"""
3. init the module, loss function and 優化器
"""
# 設備選擇 GPU 否則用 CPU 執行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SeqLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
4. 標準訓練迴圈(Training Loop)
LSTM 容易發生梯度爆炸，額外加上梯度裁剪(gradient clipping)避免權重更新失控
"""
total_epochs = 5  # 執行五次
for epoch in range(total_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度歸零
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向傳播 + 梯度裁剪 + 更新權重
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epochs_loss = running_loss / len(dataset)
    print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {epochs_loss:.4f}")
