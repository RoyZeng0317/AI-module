import math, torch, torch.nn as nn, torch.optim as optim, numpy as np

torch.manual_seed(42)
np.random.seed(42)

OPS = ["mul", "add", "sub", "transpose", "det", "inverse"]
SIZES = [2, 3]
MAX_N = max(SIZES)
IN_DIM = MAX_N * MAX_N * 2 + len(OPS) + len(SIZES)
OUT_DIM = MAX_N * MAX_N
TRAIN_PER_COMBO = 2000
TEST_PER_COMBO = 300

def pad(m):
    out = np.zeros((MAX_N, MAX_N), dtype=np.float32)
    out[:m.shape[0], :m.shape[0]] = m
    return out

# 01. build one (X, Y, mask) sample for a given op + matrix size, padded to MAX_N x MAX_N so every
# task/size shares the same input/output shape and one network can learn all of them together
def build_sample(op, n):
    A = np.random.randint(-5, 6, size=(n, n)).astype(np.float32)
    B = np.random.randint(-5, 6, size=(n, n)).astype(np.float32) if op in ("mul", "add", "sub") else np.zeros((n, n), dtype=np.float32)

    mask = np.zeros((MAX_N, MAX_N), dtype=np.float32)
    if op == "mul":
        C = A @ B; mask[:n, :n] = 1
    elif op == "add":
        C = A + B; mask[:n, :n] = 1
    elif op == "sub":
        C = A - B; mask[:n, :n] = 1
    elif op == "transpose":
        C = A.T; mask[:n, :n] = 1
    elif op == "det":
        C = np.zeros((n, n), dtype=np.float32); C[0, 0] = np.linalg.det(A); mask[0, 0] = 1
    else:  # inverse: skip singular A, it has no valid answer
        if abs(np.linalg.det(A)) < 1e-3:
            return None
        C = np.linalg.inv(A); mask[:n, :n] = 1

    op_1hot = np.eye(len(OPS), dtype=np.float32)[OPS.index(op)]
    size_1hot = np.eye(len(SIZES), dtype=np.float32)[SIZES.index(n)]
    X = np.concatenate([pad(A).flatten(), pad(B).flatten(), op_1hot, size_1hot])
    return X, pad(C).flatten(), mask.flatten()

# 02. balanced dataset: every (op, size) combination gets the same sample count
def make_dataset(per_combo):
    xs, ys, ms = [], [], []
    for op in OPS:
        for n in SIZES:
            got = 0
            while got < per_combo:
                sample = build_sample(op, n)
                if sample is None:
                    continue
                x, y, m = sample
                xs.append(x); ys.append(y); ms.append(m)
                got += 1
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys)), torch.tensor(np.array(ms))

X_train, Y_train, M_train = make_dataset(TRAIN_PER_COMBO)
X_test, Y_test, M_test = make_dataset(TEST_PER_COMBO)

# 02b. per (op, size) target scale (std of that combo's masked values) so a wide-range task like a
# 3x3 determinant can't dominate the shared loss over a small-range task like an inverse
def combo_scales(Y, M, per_combo):
    scales, idx = [], 0
    for op in OPS:
        for n in SIZES:
            vals = Y[idx:idx + per_combo][M[idx:idx + per_combo].bool()]
            scales.append(max(vals.std().item(), 1e-2))
            idx += per_combo
    return scales

S_train = torch.tensor(np.repeat(combo_scales(Y_train, M_train, TRAIN_PER_COMBO), TRAIN_PER_COMBO), dtype=torch.float32)

# 03. one shared backbone; the op/size one-hot in the input tells it which linear algebra task to run
class LinearAlgebraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(IN_DIM, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, OUT_DIM)
        )
    def forward(self, x):
        return self.network(x)

model = LinearAlgebraNet()
optimizer = optim.AdamW(model.parameters(), lr=0.01)

def masked_mse(pred, target, mask, scale=None):
    diff2 = (pred - target) ** 2
    if scale is not None:
        diff2 = diff2 / scale.unsqueeze(1) ** 2
    return (diff2 * mask).sum() / mask.sum().clamp(min=1)

# 04. training loop (loss normalized per-sample by its task's scale, evaluation stays in raw units)
epochs = 400
batch_size = 128
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train, M_train, S_train), batch_size=batch_size, shuffle=True)

print("Begin training the linear algebra multi-task model...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for bx, by, bm, bs in loader:
        optimizer.zero_grad()
        loss = masked_mse(model(bx), by, bm, bs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(loader):.6f}")

# 05. verify test set, overall then broken down per (op, size) so weak spots are visible
model.eval()
with torch.no_grad():
    pred = model(X_test)
    print(f"\nOverall Test Masked MSE: {masked_mse(pred, Y_test, M_test).item():.6f}")

    idx = 0
    for op in OPS:
        for n in SIZES:
            chunk = slice(idx, idx + TEST_PER_COMBO)
            l = masked_mse(pred[chunk], Y_test[chunk], M_test[chunk]).item()
            print(f"  {op:<10} {n}x{n} -> MSE: {l:.6f}")
            idx += TEST_PER_COMBO

# 06. random sample check per operation on a fixed 2x2 pair
sample_A = np.array([[2, 3], [-1, 4]], dtype=np.float32)
sample_B = np.array([[5, 0], [1, -2]], dtype=np.float32)

def predict(op, n=2):
    B = sample_B if op in ("mul", "add", "sub") else np.zeros((n, n), dtype=np.float32)
    op_1hot = np.eye(len(OPS), dtype=np.float32)[OPS.index(op)]
    size_1hot = np.eye(len(SIZES), dtype=np.float32)[SIZES.index(n)]
    x = torch.tensor(np.concatenate([pad(sample_A).flatten(), pad(B).flatten(), op_1hot, size_1hot]))
    with torch.no_grad():
        return model(x).numpy().reshape(MAX_N, MAX_N)[:n, :n]

real = {"mul": sample_A @ sample_B, "add": sample_A + sample_B, "sub": sample_A - sample_B,
        "transpose": sample_A.T, "inverse": np.linalg.inv(sample_A)}

print("\n--- Per-operation Testing Result (2x2 sample) ---")
print("A:\n", sample_A, "\nB:\n", sample_B)
for op in OPS:
    out = predict(op, 2)
    if op == "det":
        print(f"\n[det] Correct: {np.linalg.det(sample_A):.2f}  Predicted: {out[0, 0]:.2f}")
    else:
        print(f"\n[{op}] Correct:\n{real[op]}\nPredicted:\n{np.round(out, 2)}")
