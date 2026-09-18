import torch, torch.nn as nn, torch.nn.functional as F

class ExplicitActivationNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ExplicitActivationNet, self).__init__()
        # define Linear Layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc_binary = nn.Linear(16, 1)
        self.fc_multi = nn.Linear(16, num_classes)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))

        x2 = torch.tanh(self.fc2(x1))

        x3 = torch.sigmoid(self.fc3(x2))


        binary_output = torch.sigmoid(self.fc_binary(x3))

        multi_output = F.softmax(self.fc_multi(x3), dim=-1)

        return binary_output, multi_output

dummy_input = torch.randn(2, 10)
model = ExplicitActivationNet(input_size=10, num_classes=3)
binary_pred, multi_pred = model(dummy_input)

print("二元分類輸出(sigmoid, 範圍 0-1): \n", binary_pred)
print("多元類書出(softmax, 總和為 1): \n", multi_pred)