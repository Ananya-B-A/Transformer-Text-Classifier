# Transformer-Based Text Classification

## Overview
This project implements an end-to-end Transformer-based NLP classifier using PyTorch.
The model is built from scratch using embeddings, positional encoding, and multi-head
self-attention, and deployed as an interactive Streamlit application.

## Features
- Custom tokenization and vocabulary handling
- Transformer Encoder architecture
- Binary sentiment classification
- Model training and evaluation
- Deployment using Streamlit

## Tech Stack
- Python
- PyTorch
- Transformer Encoder
- Streamlit
- Pandas

## Project Structure
transformer-text-classifier/
├── data/
├── model.py
├── train.py
├── app.py
├── requirements.txt
└── README.md

## How to Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
