# GIRNet: Graph-Based Interference Reduction in Live Microphone Recordings

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00)
![Spektral](https://img.shields.io/badge/Spektral-1.x-0b6cfb)
![License](https://img.shields.io/badge/license-MIT-green)

GIRNet is a **time-domain multichannel** network that combines a **Wave-U-Net encoder–decoder** with a **Graph Attention (GAT) bottleneck** to reduce microphone bleed in live multitrack recordings. The graph is built from inter-channel similarity and used to suppress interference while preserving the target channel.

**Paper:** *GIRNet: A Graph-Based Interference Reduction in Live Microphone Recordings*  
Rajesh R, Padmanabhan Rajan (IIT Mandi)  
[[PDF]](paper/GIRNET.pdf) · [[Audio Demos]](https://its-rajesh.github.io/rajesh/papers/girnet)

---

## Highlights
-  **Graph Attention bottleneck** modeling inter-source relations
-  Operates on **raw waveforms** (no phase estimation)
-  Optional post-processing: **hard masks** / **spectral subtraction**
-  Simple CLI: train with NumPy `.npy` arrays for **mixtures** and **clean** targets

---

## Repository layout
```

GIRNet/
├─ GAT.py              # Model + CLI to train GIRNet
├─ getAudio.py         # Blocks/utilities used by the model
├─ CM\_Generate.py      # (Important) example for synthesizing mixtures with bleed
├─ paper/GIRNET.pdf    # Paper
└─ README.md

````

---

## Installation
```bash
# Python 3.8+ (conda recommended)
conda create -n girnet python=3.8 -y
conda activate girnet

# Install deps
pip install --upgrade pip wheel
pip install -r requirements.txt
````

> **GPU:** Install the TensorFlow build matching your CUDA/CuDnn. See TF’s compatibility table if needed.

---

## Data format

Training uses **NumPy arrays**:

* **Mixtures**  → `CM_Xtrain.npy` of shape **\[N, C, T, 1]**
* **Clean**     → `Ytrain.npy`    of shape **\[N, C, T, 1]** (or `[N, 1, T, 1]` if single target)

Where **N**: examples, **C**: channels (default 4), **T**: samples per clip.

> If your arrays are `[N, C, T]`, add the trailing channel dimension:
>
> ```python
> X = np.load('CM_Xtrain.npy')
> Y = np.load('Ytrain.npy')
> if X.ndim == 3: X = X[..., None]
> if Y.ndim == 3: Y = Y[..., None]
> np.save('CM_Xtrain.npy', X)
> np.save('Ytrain.npy', Y)
> ```

### (Important) Synthesizing mixtures

`CM_Generate.py` shows how to create room-simulated mixtures with bleed (via `pyroomacoustics`). Edit parameters inside and run:

```bash
python CM_Generate.py
```

This will create arrays you can adapt to `[N, C, T, 1]` for training.

---

## Quick start

### 1) Train

```bash
python GAT.py \
  --data_clean /path/to/Ytrain.npy \
  --data_mixed /path/to/CM_Xtrain.npy \
  --dim 220448 \
  --n_examples 4 \
  --batch_size 2 \
  --epochs 20 \
  --threshold 0.7 \
  --save_path results/girnet_model.h5 \
  --save_full_model
```

**Arguments (`GAT.py`)**

* `--data_clean` (str, required)  Path to `Ytrain.npy`
* `--data_mixed` (str, required)  Path to `CM_Xtrain.npy`
* `--dim` (int, default **220448**) Input length per example. Ideally fs*10sec.
* `--n_examples` (int, default **2**) Subset size to load (for quick runs)
* `--batch_size` (int, default **1**)
* `--epochs` (int, default **30**)
* `--threshold` (float, default **0.6**) Cosine-similarity threshold for graph adjacency
* `--save_path` (str, default `GAT_weights.h5`) Where to save model/weights
* `--save_full_model` (flag) Save the **full Keras model** (recommended for inference)

### 2) Inference / Evaluation

Use the notebook provided: [`notebooks/inference.ipynb`](notebooks/inference.ipynb)
Or minimal Python:

```python
import numpy as np
from tensorflow.keras.models import load_model
from GAT import GraphAttentionLayer  # custom layer
from spektral.layers import GATConv

# Load the full model you saved during training:
model = load_model(
    "results/girnet_model.h5",
    custom_objects={"GraphAttentionLayer": GraphAttentionLayer, "GATConv": GATConv}
)

X = np.load("CM_Xtest.npy")          # shape [N, C, T, 1] (or add ... None)
if X.ndim == 3: X = X[..., None]
Y_hat = model.predict(X, batch_size=1)
np.save("Y_hat.npy", Y_hat)
```

If you saved **weights only**, rebuild the architecture before loading weights:

```python
from GAT import build_girnet_model

model = build_girnet_model(input_shape=1024, channels=4, threshold=0.6)
model.load_weights("GAT_weights.h5")
```

---

## Tips & Notes

* **Channels (`C`)**: default is **4** in `build_girnet_model`. If your data uses a different `C`, pass it when rebuilding for inference.
* **Adjacency**: `--threshold` tunes the graph sparsity (lower ⇒ denser; higher ⇒ sparser).
* **Post-processing**: after prediction, you can apply hard masks / spectral subtraction before writing WAVs.
* **Reproducibility**: set NumPy/TF seeds and pin versions (see `requirements.txt`).

---

## Citation

If you use GIRNet in your research, please cite:

```bibtex
Under review
```

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE).




