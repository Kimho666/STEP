# STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction

## Overview

This repository contains the official implementation of the paper **[STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction]**.

STEP provides an ultra-fast, robust, two-stage action generation framework specifically designed to resolve the high inference latency of Diffusion Policies in real-world closed-loop robotic control:

1. **Spatiotemporal Consistency Predictor**: A lightweight Transformer with cross-attention that leverages current visual observations and historical actions to predict high-quality, spatiotemporally consistent initial action sequences.
2. **Velocity-aware Perturbation & Warm-Started Diffusion**: Adaptively injects perturbations to prevent real-world physical deadlocks (execution stalls), and refines the predicted action sequence using only **2~4** reverse diffusion steps, achieving performance comparable to a 100-step vanilla Diffusion Policy.

### 🌟 Key Features

- **Extreme Acceleration**: Slashes the required diffusion steps from 100 down to **2 steps** (reducing end-to-end latency to ~20ms).
- **Maintained Precision**: Outperforms both direct prediction (Action Predictor) and action reuse (Action Extrapolation) baselines. At 2 inference steps, STEP achieves an average 27.5% higher success rate than DDIM on real-world tasks.
- **Overcoming Physical Deadlocks**: Features a novel *Velocity-aware Perturbation* mechanism that flawlessly resolves static friction and execution stall issues inherent to extremely low-step real-world deployments.

## Directory Structure

```text
diffusion_policy/
├── model/
│   └── action_predictor/
│       ├── __init__.py
│       └── step_predictor_transformer.py    # STEP Spatiotemporal Predictor (Transformer)
├── policy/
│   ├── step_predictor_lowdim_policy.py      # Predictor Policy
│   ├── step_diffusion_unet_lowdim_policy.py # Diffusion Policy with Warm-Start & Perturbation
│   └── combined_inference_policy.py         # STEP Combined Inference Pipeline
├── workspace/
│   └── train_step_predictor_workspace.py    # Predictor training workspace
├── dataset/
│   └── step_action_dataset.py               # Dataset wrapper including action history (prev_action)
├── config/
│   ├── train_step_predictor_workspace.yaml
│   └── train_step_diffusion_workspace.yaml
├── load_official_weights.py                 # Seamless loader for official/existing weights ⭐
├── demo_combined_inference.py               # Demo script
└── eval_combined_inference.py               # Evaluation script
```

## Quick Start

## Environment Setup (Important)

STEP uses the **same software environment** as the original Diffusion Policy repository.
If you already have a working Diffusion Policy environment, you can use it directly.

Recommended setup:

```bash
# Create environment (same dependency spec as Diffusion Policy)
conda env create -f conda_environment.yaml

# Activate
conda activate diffusion_policy

# Install this repo in editable mode
pip install -e .
```

For platform-specific setups, you can also use:

- `conda_environment_real.yaml` (real-robot stack)
- `conda_environment_macos.yaml` (macOS)

This design keeps STEP drop-in compatible with existing Diffusion Policy training/evaluation workflows.

### Method 1: Using Official/Existing Diffusion Weights (Recommended)

STEP is entirely plug-and-play. If you already have official pre-trained Diffusion Policy weights, you can seamlessly load and accelerate them:

```bash
# Convert vanilla 100-step weights to STEP format (supports warm-start from predicted trajectory, requiring only 2 steps)
python load_official_weights.py --ckpt path/to/official_checkpoint.ckpt --convert_to_step --init_steps 2
```

#### Python API Usage

```python
from load_official_weights import load_official_as_step, create_step_combined_policy

# Method 1: Load and convert to STEP warm-start version (K' = 2 steps)
step_policy, cfg = load_official_as_step(
    'path/to/official.ckpt',
    init_trajectory_steps=2 
)

# Method 2: Create a combined strategy using existing Diffusion weights + a trained STEP Predictor
combined_step = create_step_combined_policy(
    diffusion_ckpt='path/to/official.ckpt',
    predictor_ckpt='path/to/step_predictor.ckpt'
)
```

### Method 2: Training from Scratch

### 1. Train the STEP Predictor (Spatiotemporal Consistency Predictor)

```bash
# Train the predictor using MSE Loss
python train.py --config-name=train_step_predictor_workspace
```

### 2. Train the Diffusion Policy

```bash
# Train the underlying conditional Diffusion model
python train.py --config-name=train_step_diffusion_workspace
```

### 3. STEP Combined Inference & Evaluation

```bash
# Run combined evaluation (automatically enables spatiotemporal warm-start & velocity-aware perturbation)
python eval_combined_inference.py \
    --predictor_ckpt path/to/step_predictor.ckpt \
    --diffusion_ckpt path/to/diffusion_policy.ckpt \
    --inference_steps 2 \
    --output_dir data/outputs/eval_step \
    --n_test 50
```

## Configuration (Aligned with Paper Table 7)

### STEP Predictor Configuration

Key hyperparameters:
- `action_chunk_size`: 16
- `hidden_dim`: 128 (Transformer embedding dimension)
- `n_layer`: 2 (Number of Cross-attention blocks)
- `n_obs`: 2 (Number of visual observation frames)

### STEP Inference & Perturbation Configuration

Key parameters:
- `total_diffusion_steps`: 100
- `warm_start_steps` ($K'$): Recommended **2** or 4
- `sigma_scale`: Predictor retention ratio during normal execution.
- `sigma_stall`: Perturbation magnitude to prevent physical deadlocks (Recommended: 0.1 for simulation, 1.0~1.4 for real-world setups).

## STEP Inference Pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│              STEP Real-time Closed-Loop Inference               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Current Obs  │ (o_t)                                         │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────┐     ┌──────────────────┐              │
│  │ Spatiotemporal       │◄────│ Past Actions     │ (A_{t-H})    │
│  │ Predictor (2-layer)  │     │ (Temp. Smoothness│              │
│  └──────────┬───────────┘     └──────────────────┘              │
│             │                                                   │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │ Initial Prediction   │ (\hat{A}_t)                           │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│  ┌──────────────────────┐      Execution Stall? (||ΔA_t|| < ε)  │
│  │ Velocity-aware       │ ──── Yes ──> Inject Noise (σ_stall)   │
│  │ Perturbation         │ ──── No ───> Scale Action (σ_scale)   │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│             ▼ Warm-start initialization (K'=2)                  │
│  ┌──────────────────────┐                                       │
│  │ Warm-started         │                                       │
│  │ Diffusion Policy     │ (Only 2 steps to reach target mode)   │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │ Final Action A_t     │────────────┐                          │
│  └──────────────────────┘            │                          │
│                                      │ Cache Action             │
│                                      ▼                          │
│                           ┌──────────────────┐                  │
│                           │ Update A_{cache} │                  │
│                           └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## Python API Example

```python
from diffusion_policy.policy.combined_inference_policy import STEPCombinedPolicy

# Initialize STEP Policy
step_policy = STEPCombinedPolicy(
    predictor_ckpt='path/to/ap.ckpt',
    diffusion_ckpt='path/to/dp.ckpt',
    device='cuda:0',
    warm_start_steps=2,
    use_velocity_perturbation=True, # Prevent real-world execution deadlocks
    stall_threshold=0.01
)

# Closed-loop inference
obs_dict = {'obs': obs_tensor}  #[B, n_obs_steps, obs_dim]
step_policy.reset()

# Obtain high-quality actions in ~20ms
result = step_policy.predict_action(obs_dict)
action = result['action'] 
```

## Experimental Comparison (Excerpt from RoboMimic - ToolHang)

ToolHang is one of the most challenging contact-rich tasks in the RoboMimic benchmark. Under aggressive step reduction ($K=2$), existing acceleration methods completely fail, whereas STEP maintains strong generalization and multimodality.

| Method | Denoising Steps | Inference Latency | Success Rate |
|------|----------|----------|------|
| Vanilla DDPM | 100 | ~674 ms | 0.68 (Upper Bound) |
| DDIM | 2 | ~16 ms | 0.06 (Collapses) |
| BRIDGER | 2 | ~15 ms | 0.08 (Collapses) |
| **STEP (Ours)** | **2** | **~19 ms** | **0.64 (Near Upper Bound)** |

*STEP drastically improves success rates at just 2 denoising steps with negligible latency overhead, truly unlocking high-frequency, real-time complex visuomotor control.*