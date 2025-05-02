# 🧠 MNIST Transformer Project

This repository explores the MNIST dataset using Transformer-based architectures in two distinct ways:

---

## 📌 Project 1: Encoder-only Vision Transformer (ViT)

**File:** `ViT_Encoder_Classification.ipynb`

A Vision Transformer classifier trained directly on the MNIST dataset using only the encoder part of the Transformer. Each image is split into patches, linearly projected, and passed through standard Transformer encoder blocks to classify the digit.

---

## 📌 Project 2: Encoder–Decoder Transformer for Tiled Digit Recognition

**Files:**
- `train_ViT.py`
- `model.py`
- `mnist_generator.py`
- `ViT_Encoder_Decoder.ipynb`

We create synthetic sequences by tiling multiple MNIST digits (2x2 grid) into a single image. The encoder processes the image, and the decoder generates the digit sequence using teacher forcing and greedy decoding.

---

## 🧪 Data Generation

**File:** `mnist_generator.py`

Generates:
- Standard MNIST samples  
- Tiled 2×2 digit sequences  
- Optional blanks for variable-length sequence generation

Classes:
- `MNISTDataset`
- `TiledMNISTDataset`
- `ScatteredMNISTDataset`

---

## 🧠 Model Architectures

**File:** `model.py`

Contains:
- `PatchEmbedding`
- `MultiHeadAttention`, `AttentionHead`
- `TransformerBlock`, `TransformerMNIST`
- Greedy decoding logic

---

## 🧰 Project Structure

