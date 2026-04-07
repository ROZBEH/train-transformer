# Transformer from Scratch — TinyShakespeare

A decoder-only, GPT-style Transformer implemented from scratch in PyTorch and trained character-by-character on the [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset.

## Architecture

- **Multi-head self-attention** with causal masking and optional KV caching for fast inference
- **Sinusoidal positional encoding**
- **6 decoder layers**, each with pre-norm LayerNorm, MHA, GELU MLP (4× expansion), and residual connections
- **Character-level tokenizer** with `<start>`, `<end>`, and `<pad>` special tokens
- **Beam search** and **top-k sampling** generation strategies

## Project Structure

```
train-transformer/
├── transformer.py        # Model definition (Transformer, Decoder, MHA, KV cache)
├── utils.py              # Tokenizer, data prep, text generation utilities
├── train_transformer.py  # Training & evaluation CLI script
├── tinyshakespeare.txt   # Training corpus
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** PyTorch will automatically use the MPS backend. On other hardware it falls back to CPU. For CUDA GPU support install the appropriate PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

### Train

```bash
python train_transformer.py --train
```

This trains for 10 epochs and saves two checkpoint files:
- `model_checkpoint.pth` — full checkpoint (model + optimizer state)
- `model_state_dict.pth` — model weights only (used for eval)

### Evaluate / Generate text

```bash
python train_transformer.py --eval
```

Loads `model_state_dict.pth` and generates continuations for two seed phrases using beam search:
- `"I speak from"`
- `"Some parcels of"`

### Train + Eval in one go

```bash
python train_transformer.py --train --eval
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence length | 40 |
| Batch size | 32 |
| Model dimension (`d_model`) | 512 |
| Attention heads | 8 |
| Decoder layers | 6 |
| Learning rate | 5e-4 |
| Epochs | 10 |
| LR schedule | OneCycleLR |
| Label smoothing | 0.02 |
| Gradient clipping | 1.0 |
