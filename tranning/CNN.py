import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""
1. Denfine the CNN module
"""
class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()

        # 卷基層 1: 輸入 1 通道(灰階圖)，輸出 16 通道，核心 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸減半

        # 卷機層 2: 輸入 16 通道，輸出 32 通道，核心 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸再減半

        # 全連接分類器 (假設輸入團圖片大小為 28x28，經過兩市 MaxPool 後變為 7x7)
        self.fc = nn.Linear(32 * 7 * 7 , num_class)
    def forward(self, x):
        # x shape: (Batch, Channel, Height, Width)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # 展平 Tensor (Flatten) 以輸入全連接層
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
"""
2. 模擬數據與 DataLoader setup
模擬 100 張 1x28x28 的圖片與對應標籤
"""
dummy_x = torch.randn(100, 1, 28, 28)
dummy_y = torch.randint(0, 10, (100,))

dataset = TensorDataset(dummy_x, dummy_y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

"""
3. init the module, loss and function and 優化器
"""
# 設備選擇 GPU 否則用 CPU 執行，先用 GPU 跑且沒有改用 CPU，GPU 速度會比 CPU 快很多
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_class=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
4. 標準訓練迴圈(Tranning Loop)
"""
total_epochs = 5  # 執行五次
for epoch in range(total_epochs):
    model.train()   # switch to the tranning mode
    running_loss = 0.0  # 目標損失函數值為 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度歸零
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向傳播 + 更新權重
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epochs_loss = running_loss / len(dataset)
    print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {epochs_loss:.4f}")