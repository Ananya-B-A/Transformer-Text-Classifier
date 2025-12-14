import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from model import TransformerClassifier

# ------------------ Load data ------------------
df = pd.read_csv("data/data.csv")

# Special tokens
vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2
}

def encode(sentence):
    tokens = sentence.lower().split()
    encoded = [vocab["<CLS>"]]

    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
        encoded.append(vocab[t])

    return encoded

X = [encode(s) for s in df["text"]]
y = torch.tensor(df["label"].values)

# ------------------ Padding ------------------
MAX_LEN = 50

X = [x[:MAX_LEN] + [0] * (MAX_LEN - len(x)) for x in X]
X = torch.tensor(X)

# Padding mask (True = ignore)
padding_mask = (X == 0)

# ------------------ Save vocab ------------------
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

# ------------------ Model ------------------
model = TransformerClassifier(len(vocab), MAX_LEN)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# ------------------ Training ------------------
model.train()

for epoch in range(80):
    optimizer.zero_grad()

    logits = model(X, mask=padding_mask)
    loss = criterion(logits, y.float())

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ------------------ Save model ------------------
torch.save(model.state_dict(), "model.pth")
print("✅ Training complete.")
