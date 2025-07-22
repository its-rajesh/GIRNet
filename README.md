# GIRNet: Graph-Based Interference Reduction in Live Microphone Recordings

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

GIRNet is a neural network architecture designed for interference reduction in live multitrack audio recordings, particularly where microphone bleed is a challenge. It combines a Wave-U-Net encoder-decoder with a Graph Attention Network (GAT) bottleneck to model inter-source relationships.

This repository contains the official implementation for the paper:

**GIRNet: A Graph-Based Interference Reduction in Live Microphone Recordings**  
Rajesh R (UIC) and Padmanabhan Rajan (IIT Mandi)  
[[Paper]](paper/GIRNET.pdf) | [[Audio Demos]](https://its-rajesh.github.io/rajesh/papers/girnet)

---

## Key Features

- Time-domain multichannel interference reduction
- Learnable Graph Attention-based bottleneck
- Works on raw waveforms, no phase estimation required
- Postprocessing with hard masks and spectral subtraction

---

## Folder Overview

- `scripts/`: Main training and testing code (GIRNet, GAT, data pipeline)
- `data/`: Add your dataset here. Expects waveform `.wav` files.
- `results/`: Stores model predictions and logs
- `notebooks/`: Jupyter notebooks for visualization and testing
- `paper/`: PDF of the published manuscript

---

## Quick Start

### 1. Install dependencies

```bash
conda create -n girnet python=3.8
conda activate girnet
pip install -r requirements.txt
```

### 2. Prepare dataset (For Training)


### 3. Training

```
python CM_Generate.py \
  --data_clean /path/to/Ytrain.npy \
  --data_mixed /path/to/CM_Xtrain.npy \
  --dim 1024 \
  --n_examples 4 \
  --batch_size 2 \
  --epochs 20 \
  --threshold 0.7 \
  --save_path results/girnet_model.h5 \
  --save_full_model
```
