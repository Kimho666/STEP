# Diffusion Policy + Action Predictor Instructions

## Architecture Snapshot
- **Hybrid System**: A two-stage inference system where `ActionPredictor` (Transformer) generates an initial trajectory, and `EnhancedDiffusionUnetPolicy` (Diffusion) refines it. See `diffusion_policy/policy/combined_inference_policy.py`.
- **Decoupled Components**: Tasks (datasets/runners) and methods (policies/workspaces) are separated. 
- **Workspace-Driven**: `Workspace` objects (in `diffusion_policy/workspace/`) orchestrate training. `train.py` uses Hydra to instantiate a workspace and calls `workspace.run()`.
- **Initialization**: `Enhanced` policies support `init_trajectory`, allowing diffusion to start from partial noise (e.g., 25 steps instead of 100).

## Generative Methods
| Method | Policy | Config | Inference Steps |
|--------|--------|--------|-----------------|
| Diffusion (DDPM) | `DiffusionUnetLowdimPolicy` | `train_diffusion_unet_lowdim_workspace` | 100 |
| **CFM (OT-CFM)** | `CFMUnetLowdimPolicy` | `train_cfm_unet_lowdim_workspace` | **10** (or 2-4 fast) |

### CFM Baseline (Flow Matching)
- **No noise scheduler**: CFM predicts velocity `v = x_1 - x_0` directly.
- **Linear path**: `x_t = (1-t)*x_0 + t*x_1` (Optimal Transport).
- **Euler solver**: Simple `x_{t+dt} = x_t + v*dt` integration.
- **Time embedding**: Continuous `t ∈ [0,1]` scaled to `[0,1000]` for sinusoidal compatibility.

## Core Workflows
- **Setup**: `conda activate robodiff`, then `pip install -e .`.
- **Training**: Run `python train.py --config-name=<workspace_name>`. 
    - Action Predictor: `train_action_predictor_lowdim_workspace`.
    - Diffusion: `train_diffusion_unet_lowdim_workspace` or `train_enhanced_diffusion_unet_lowdim_workspace`.
    - **CFM**: `train_cfm_unet_lowdim_workspace` (10-step baseline).
- **Inference**: Use `demo_combined_inference.py` for testing the full pipeline.
- **Weights**: Use `load_official_weights.py` to convert official checkpoints to the `Enhanced` format.

## Project Patterns & Conventions
- **Hydra Everything**: Configurations are in `diffusion_policy/config/`. Use Hydra overrides for experiments (e.g., `training.device=cuda:0`).
- **Normalizers**: Policies require a `LinearNormalizer`. Datasets must provide one. Key alignment is critical.
- **Environment Runners**: Inherit from `BaseImageRunner` or `BaseLowdimRunner`. They must return dicts compatible with `wandb` logging.
- **Zarr Datasets**: Large-scale data is often stored as Zarr; see `diffusion_policy/common/replay_buffer.py`.

## Key Files
- `diffusion_policy/policy/cfm_unet_lowdim_policy.py`: **CFM baseline** with Euler solver.
- `diffusion_policy/policy/enhanced_diffusion_unet_lowdim_policy.py`: The core refinement logic.
- `diffusion_policy/model/action_predictor/action_predictor_transformer.py`: The initial trajectory generator.
- `ACTION_PREDICTOR_README.md`: Detailed system overview.
- `COMMANDS_GUIDE.md`: Canonical command list.

## Debugging Tips
- Check `data/outputs/<timestamp>/logs.json.txt` for training logs.
- Use `training.debug=True` in Hydra overrides for faster iteration.
- Unit tests in `tests/` cover core utilities like `replay_buffer` and `cv2_util`.

