import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

# 01. read the hanle the JSON dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
json_file = os.path.normpath(os.path.join(current_dir, "mental_health_data.json"))

with open(json_file, 'r', encoding='utf-8') as f:
    dataset_json = json.load(f)

prompts = [item['prompt'] for item in dataset_json]
#  reply and emption to label

labels = [item.get('score', 0) for item in dataset_json]

def build_vocab(texts, max_vocab_size=1000):
    all_chars = "".join(texts)
    char_counts = Counter(all_chars)
    most_common = char_counts.most_common(max_vocab_size - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char, _ in most_common:
        vocab[char] = len(vocab)
    return vocab

vocab = build_vocab(prompts)
max_len = 30

def sequence(text, vocab, max_len):
    seq = [vocab.get(char, vocab["<UNK>"]) for char in text[:max_len]]
    seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
    return seq
x = [sequence(p, vocab, max_len) for p in prompts]
y = labels

# cut traning set (80%) verify set (20%) -- this is avoid the always to remember it of keypoint.
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

class TextDataSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_loader = DataLoader(TextDataSet(x_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TextDataSet(x_val, y_val), batch_size=16, shuffle=False)

# 02. design the avoid and self create keneral network model
class GeneralizingEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_p=0.3):
        super(GeneralizingEmotionClassifier, self).__init__()
        # 01. Embedding : create the  space, avoid the always remember
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 02. 
        self.fc1 = nn.Linear(max_len * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()

        # 03. Dropout:
        self.dropout = nn.Dropout(p=dropout_p)

        # 04. output and Softmax (multiple types to category)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)       # (Batch, Max_Len, Embed_Dim)
        x = x.view(x.size(0), -1)   #  matrix
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)         # when tranning random to forget it
        out = self.fc2(x)           # Output the Logits  (match the CrossEntropyLoss inside auto  softmax)
        return out

model = GeneralizingEmotionClassifier(
    vocab_size=len(vocab),
    embedding_dim=32,
    hidden_dim=16,      # zoom out the hidden_dim, for model to learning
    num_classes=6,      # action.py score 是 0~5 六級
    dropout_p=0.4       # 40% Dropout avoid to always remember
)

criterion = nn.CrossEntropyLoss()
# add Weight Decay (L2 are  numpy), avoid the weight pretty big
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# 03. while to tranning and Early Stopping
epochs = 50
best_val_loss = float('inf')
patience = 5        # verify contiune loss are stopping
patience_counter = 0

print("Starting to trainning (and avoid )...")
for epoch in range(epochs):
    # --- tranning step ---
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- verify step(use to monitor is always to remember or not.) ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}]")

    # --- Early Stopping logi ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        #  save the best  model weight
        torch.save(model.state_dict(), os.path.join(current_dir, "best_model.pth"))

    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Check the loss are stoppping verify are going down (model are start to always to ), Early Stopping !")
            break
print("Finished are tranning! Dropout, L2 weight are going loss, Ealy Stop has save model is good  abilirt.")