# Multi-Modal Data-Driven Velocity Model Building with Auxiliary Seismic Images Based on OT-CFM

Official implementation of the paper:

> **A Multi-Modal-Data-Driven Velocity Model Building Method with Auxiliary Seismic Images Based on Optimal Transport Conditional Flow Matching**
> Shuliang Wu, and Jianhua Geng
> *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*

---

## Overview

We propose a data-driven, multi-task learning method for seismic velocity model building. The method uses **Optimal Transport Conditional Flow Matching (OT-CFM)** to simultaneously generate high-resolution velocity models and auxiliary seismic images, conditioned on three modalities of input data:

- Multi-trace CMP (Common Midpoint) gathers
- Smooth velocity model
- Synthetic seismic image derived from the smooth model via the convolution model

A key innovation is the **physics-informed consistency loss**, which explicitly couples the predicted velocity model and seismic image through the convolution model in the loss function — mirroring the relationship exploited by conventional methods such as FWI and RWI.

![Method Diagram](media/image3.png)

**Key advantages over existing methods:**
- Avoids spectral bias of standard CNNs/Transformers via generative flow matching
- Faster inference than diffusion models: **1 second** per full velocity model (vs. 7s for DM, 20 hours for RWI)
- Only 20 sampling steps (vs. 200+ for diffusion models)
- Local multi-trace processing reduces dimensionality and allows inversion of 2D models of arbitrary horizontal extent

---

## Method

### OT-CFM Training Objective

The network learns to predict the velocity field between noise and target:

$$\mathcal{L}_{OT-CFM} = \left\| v_\theta(t, \dot{x}, z) - (x_1 - x_0) \right\|^2$$

where $x_1 = (img, vel)$ is the joint target (image + velocity), $x_0 \sim \mathcal{N}(0,I)$, and the interpolated state is $\dot{x} = t \cdot x_1 + (1-t) \cdot x_0$.

The condition $z = (data_{CMP},\ img_{smooth},\ vel_{smooth})$ is processed by 1D convolution layers and a CBAM attention block before being added to the sinusoidal time embedding.

### Physics Consistency Loss

The predicted velocity $vel_\theta$ is used to compute synthetic reflectivity:

$$R_\theta = \frac{vel_\theta^i - vel_\theta^{i-1}}{vel_\theta^i + vel_\theta^{i-1}}$$

and then convolved with the Ricker wavelet to obtain $\widetilde{img}_\theta$. The consistency loss is:

$$\mathcal{L}_{consistency} = MSE(\widetilde{img}_\theta, img_\theta) + MSE(\widetilde{img}_\theta, img) + SSIM(\widetilde{img}_\theta, img_\theta) + SSIM(\widetilde{img}_\theta, img)$$

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{OT-CFM} + \mathcal{L}_{consistency}$.

### Network Architecture

A modified **conditional U-Net** with:
- 3 down-sampling blocks (MaxPool2D + DoubleConv + SelfAttention)
- 3 up-sampling blocks (Upsample + DoubleConv + SelfAttention)
- Sinusoidal time embedding injected into every down/up block
- Input conditioning via parallel 1D Conv branches + CBAM attention fusion

### Multi-Trace Local Processing

The model generates 8-trace 1D velocity profiles. Overlapping traces between adjacent blocks are averaged to produce the final 2D model. This strategy:
1. Reduces the mapping complexity from 2D to near-1D
2. Allows inference on 2D models of any horizontal extent
3. Reduces the need for large labeled 2D datasets

---

## Results

### Synthetic Data (OpenFWI)

| Method | FlatB RMSE (m/s) | FlatB SSIM | FlatfaultB RMSE | FlatfaultB SSIM |
|--------|-----------------|------------|-----------------|-----------------|
| U-Net  | 904.495         | 0.194      | 969.247         | 0.329           |
| ViT    | 717.577         | 0.358      | 744.948         | 0.452           |
| DM     | 178.540         | 0.705      | 132.174         | 0.784           |
| OT-CFM | 56.947          | 0.944      | 69.292          | 0.942           |
| **Proposed** | **32.709** | **0.985** | **33.854**     | **0.959**       |

Training: 200 epochs, ~8 hours on a single RTX 3080Ti (12GB).
Inference: **~1 second** for a full velocity model.

### Generalization

| Test Set     | Without Fine-tuning SSIM | With 10% Fine-tuning SSIM |
|--------------|--------------------------|---------------------------|
| Marmousi2    | 0.842                    | **0.950**                 |
| Overthrust   | 0.693                    | **0.935**                 |

### Field Data (East China Sea)

Applied to marine towed-streamer data (486 hydrophones, 450 shots, 17.5 km × 3.9 km). The method generates a velocity model in **3 seconds on a single GPU**, compared to **20 hours** for Reflection Waveform Inversion (RWI) on four GPUs. ADCIGs from the generated velocity model show superior flatness and resolution compared to RWI.

---

## Repository Structure

```
.
├── CFM-MULTI-TASK-VMB.ipynb          # Training pipeline (data loading, training loop)
├── CFM-FIELD-DATA.ipynb              # Inference on field data
├── nn_for_fwi_80_8_attention_3.py    # Network architecture (UNet_conditional, EMA)
└── utils.py                          # Utilities (not included; see dependencies below)
```

**`nn_for_fwi_80_8_attention_3.py`** defines:
- `UNet_conditional` — the main conditional U-Net backbone
- `EMA` — Exponential Moving Average for stable inference-time model

**`CFM-MULTI-TASK-VMB.ipynb`** contains:
- `RFlow` — Rectified Flow sampling/training logic
- `MSSSIMLoss` — Multi-scale SSIM loss
- `generate_migrated_data_cuda` — GPU-accelerated synthetic seismic generation (convolution model)
- `make_vp_net` — Multi-trace data packaging
- Full training loop with dynamic gradient-norm-based task weighting

---

## Data

### Training Data (OpenFWI)

| Subset       | # Models | Notes                     |
|--------------|----------|---------------------------|
| FlatfaultB   | 800      | from [OpenFWI](https://github.com/lanl/OpenFWI) |
| CurvelB      | 400      | from OpenFWI              |
| FlatB        | 200      | from OpenFWI              |

Original grid: 64×64 points. Expanded to **188×64** (3.76 km × 1.28 km, DX = 20 m), yielding 263,200 training pairs.

CMP data are simulated with the **scalar acoustic wave equation**, 12 Hz Ricker wavelet, 2 ms time sampling, 4 s record length.

Imaging labels are synthesized by convolving true velocity models with a **30 Hz Ricker wavelet**. Smooth velocity models are obtained by Gaussian smoothing (σ = 2).

Expected data layout on the server:
```
/data/wsl/model_openfwi/
├── flatfault/
│   ├── model_200_70/           # model{i}.bin  [NZ=200, NX=70] float32
│   └── cmp_data_200_70_add_layer/{i}_cmp/
├── curvevelB/
│   └── ...
└── flat_b/
    └── ...
```

### Field Data

```
/data/wsl/jinshan_data/
├── filed_cmp_1260_2_650_cut    # CMP gathers  [1260, 2, 650] float32
├── vel.bin                     # Initial velocity [1360, 250] float32
├── cmp_net.npy                 # Pre-processed CMP
└── rtm_net.npy                 # Pre-processed RTM image
```

---

## Training

Requirements:
```
torch torchvision tqdm tensorboard numpy scipy matplotlib Pillow
```

GPU: CUDA-enabled GPU with ≥12 GB VRAM (tested on NVIDIA RTX 3080Ti).

Run cells sequentially in `CFM-MULTI-TASK-VMB.ipynb`:
1. **Cells 1–4**: imports, loss functions, utility functions
2. **Cell 5**: load and preprocess data (reads `.bin` files from `/data/wsl/model_openfwi/`)
3. **Cell 6**: normalize and package into training arrays
4. **Cell 7**: initialize model, optimizer, scheduler
5. **Cell 8**: training loop (300 epochs by default)

Checkpoints are saved to `models/<run_name>/`:
- `{epoch}_ckpt.pt` every 100 epochs
- `ckpt.pt` latest weights
- `ckpt_ema.pt` EMA model weights (recommended for inference)

Key hyperparameters (set in Cell 6–7):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_channel` | 8 | Number of neighboring traces per sample |
| `img_size` | 64 | Spatial output resolution |
| `wavelet_freq` | 30 Hz | Ricker wavelet frequency for imaging labels |
| `smooth_sigma` | 5 | Gaussian smoothing σ for initial velocity |
| `batch_size` | 400 | Training batch size |
| `lr` | 3e-4 | AdamW learning rate |
| `epochs` | 300 | Total training epochs |
| `step` (RFlow) | 20 | Euler integration steps at inference |

---

## Inference

Run `CFM-FIELD-DATA.ipynb` for field data inference. Load the trained EMA checkpoint and call `rf.sample_for_gen()` with the three conditioning inputs. Adjacent 8-trace output blocks are averaged on overlapping traces to reconstruct the full 2D model.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wu2025otcfm,
  title   = {A Multi-Modal-Data-Driven Velocity Model Building Method with Auxiliary Seismic Images Based on Optimal Transport Conditional Flow Matching},
  author  = {Wu, Shuliang and Geng, Jianhua},
  year    = {2026}
}
```

---

## Acknowledgements

This research is supported by the National Natural Science Foundation of China under Grant 42330805.

Shuliang Wu and Jianhua Geng are with the Shanghai Key Laboratory of Submarine Clean Resources Exploration and Development, School of Ocean and Earth Science, Tongji University. (Contact: jhgeng@tongji.edu.cn)
