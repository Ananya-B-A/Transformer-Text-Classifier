import streamlit as st
import torch
import pickle
from model import TransformerClassifier

st.title("Transformer Text Classifier")

# ------------------ Load vocab ------------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

MAX_LEN = 50

# ------------------ Load model ------------------
model = TransformerClassifier(len(vocab), MAX_LEN)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

text = st.text_input("Enter text:")

if text:
    tokens = [vocab["<CLS>"]]
    for w in text.lower().split():
        tokens.append(vocab.get(w, vocab["<UNK>"]))

    tokens = tokens[:MAX_LEN] + [0] * (MAX_LEN - len(tokens))
    x = torch.tensor(tokens).unsqueeze(0)

    mask = (x == 0)

    with torch.no_grad():
        logits = model(x, mask=mask)
        prob = torch.sigmoid(logits).item()

    st.write("Confidence:", round(prob, 3))
    st.write(
        "Prediction:",
        "Positive 😊" if prob > 0.55 else "Negative 😞"
    )
