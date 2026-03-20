import time
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/khanhnd/HPC/project-3-winter-2026-khanh14ph/milestone1/data/"
EPOCHS     = 50
BATCH_SIZE = 500
LR         = 0.01
TRAIN_SIZE = 50_000
VAL_SIZE   = 10_000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MNIST loader ──────────────────────────────────────────────────────────────
def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(np.float32) / 255.0

def load_mnist_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        # He initialization (matches C++ implementation)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # Load data
    train_images = load_mnist_images(DATA_DIR + "train-images-idx3-ubyte")
    train_labels = load_mnist_labels(DATA_DIR + "train-labels-idx1-ubyte")
    test_images  = load_mnist_images(DATA_DIR + "t10k-images-idx3-ubyte")
    test_labels  = load_mnist_labels(DATA_DIR + "t10k-labels-idx1-ubyte")

    X_train = torch.from_numpy(train_images[:TRAIN_SIZE])
    y_train = torch.from_numpy(train_labels[:TRAIN_SIZE])
    X_val   = torch.from_numpy(train_images[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE])
    y_val   = torch.from_numpy(train_labels[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE])
    X_test  = torch.from_numpy(test_images)
    y_test  = torch.from_numpy(test_labels)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=BATCH_SIZE)

    model     = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    total_samples = 0
    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * X_batch.size(0)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            total_samples += X_batch.size(0)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_correct += (model(X_batch).argmax(1) == y_batch).sum().item()

        avg_loss   = train_loss / TRAIN_SIZE
        train_acc  = 100.0 * train_correct / TRAIN_SIZE
        val_acc    = 100.0 * val_correct   / VAL_SIZE
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"loss={avg_loss:.4f}  "
              f"train={train_acc:.2f}%  "
              f"val={val_acc:.2f}%")

    t_end    = time.perf_counter()
    elapsed  = t_end - t_start
    grind    = total_samples / elapsed

    # ── Test ──────────────────────────────────────────────────────────────────
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            test_correct += (model(X_batch).argmax(1) == y_batch).sum().item()
    test_acc = 100.0 * test_correct / len(y_test)

    print(f"\nTest accuracy : {test_acc:.2f}%")
    print(f"Training time : {elapsed:.2f} s")
    print(f"Grind rate    : {grind:,.0f} samples/s")

if __name__ == "__main__":
    main()
