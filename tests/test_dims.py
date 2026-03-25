import sys
import os
import pathlib
import torch
import hydra
from omegaconf import OmegaConf
import numpy as np

# 添加项目根目录到 sys.path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(ROOT_DIR)

# 尝试导入加载函数
try:
    from eval_combined_inference import load_action_predictor, load_diffusion_policy
except ImportError:
    print("Error: Could not import from eval_combined_inference.py. Make sure it exists in the parent directory.")
    sys.exit(1)

def test_dimensions(action_predictor_ckpt, diffusion_policy_ckpt, device='cuda:0'):
    print(f"==================================================")
    print(f"Testing Dimensions Compatibility")
    print(f"==================================================")
    print(f"Action Predictor Checkpoint: {action_predictor_ckpt}")
    print(f"Diffusion Policy Checkpoint: {diffusion_policy_ckpt}")
    print(f"Device: {device}")
    
    # 1. 加载模型
    print(f"\n[1] Loading Models...")
    try:
        ap_policy, ap_cfg = load_action_predictor(action_predictor_ckpt, device)
        print(f"✅ Action Predictor loaded.")
    except Exception as e:
        print(f"❌ Failed to load Action Predictor: {e}")
        return

    try:
        dp_policy, dp_cfg = load_diffusion_policy(diffusion_policy_ckpt, device)
        print(f"✅ Diffusion Policy loaded.")
    except Exception as e:
        print(f"❌ Failed to load Diffusion Policy: {e}")
        return

    # 2. 检查静态属性
    print(f"\n[2] Checking Static Attributes...")
    
    # 获取属性
    ap_horizon = ap_policy.horizon
    dp_horizon = dp_policy.horizon
    
    ap_action_dim = ap_policy.action_dim
    dp_action_dim = dp_policy.action_dim
    
    ap_obs_dim = ap_policy.obs_dim
    dp_obs_dim = dp_policy.obs_dim

    # 打印对比
    print(f"{'Attribute':<15} | {'Action Predictor':<20} | {'Diffusion Policy':<20} | {'Status'}")
    print("-" * 70)
    
    # Horizon Check
    status = "✅ Match" if ap_horizon == dp_horizon else "⚠️ Mismatch (Handled)"
    print(f"{'Horizon':<15} | {ap_horizon:<20} | {dp_horizon:<20} | {status}")
    
    # Action Dim Check
    status = "✅ Match" if ap_action_dim == dp_action_dim else "❌ Mismatch (Critical)"
    print(f"{'Action Dim':<15} | {ap_action_dim:<20} | {dp_action_dim:<20} | {status}")
    
    # Obs Dim Check
    status = "✅ Match" if ap_obs_dim == dp_obs_dim else "❌ Mismatch (Critical)"
    print(f"{'Obs Dim':<15} | {ap_obs_dim:<20} | {dp_obs_dim:<20} | {status}")

    if ap_action_dim != dp_action_dim or ap_obs_dim != dp_obs_dim:
        print("\n❌ Critical dimension mismatch detected! Inference will likely fail.")
        return

    # 3. 运行时测试
    print(f"\n[3] Running Runtime Inference Test...")
    
    # 创建 Dummy Input
    B = 1
    n_obs_steps = ap_policy.n_obs_steps
    obs_dim = ap_policy.obs_dim
    
    dummy_obs = torch.randn((B, n_obs_steps, obs_dim), device=device)
    obs_dict = {'obs': dummy_obs}
    
    print(f"Created dummy observation: {dummy_obs.shape}")
    
    # Step A: Action Predictor Inference
    print("\nRunning Action Predictor...")
    try:
        with torch.no_grad():
            ap_result = ap_policy.predict_action(obs_dict)
            action_pred = ap_result['action_pred']
        print(f"✅ Success. Output shape: {action_pred.shape}")
    except Exception as e:
        print(f"❌ Action Predictor inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step B: Diffusion Policy Inference (with init_action)
    print("\nRunning Diffusion Policy (with init_action)...")
    try:
        with torch.no_grad():
            # 这里模拟 CombinedInferencePolicy 的行为
            dp_result = dp_policy.predict_action(obs_dict, init_action=action_pred)
            final_action = dp_result['action']
        print(f"✅ Success. Output shape: {final_action.shape}")
        print(f"\n🎉 Test Passed! The models are compatible for combined inference.")
        
        if ap_horizon != dp_horizon:
            print(f"Note: Horizon mismatch ({ap_horizon} vs {dp_horizon}) was successfully handled by the policy code.")
            
    except Exception as e:
        print(f"❌ Diffusion Policy inference failed: {e}")
        print("\nPossible causes:")
        print("1. Shape mismatch that wasn't handled.")
        print("2. Device mismatch.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_predictor_ckpt', type=str, required=True, help='Path to Action Predictor checkpoint')
    parser.add_argument('--diffusion_policy_ckpt', type=str, required=True, help='Path to Diffusion Policy checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.action_predictor_ckpt):
        print(f"Error: Action Predictor checkpoint not found: {args.action_predictor_ckpt}")
        sys.exit(1)
    if not os.path.exists(args.diffusion_policy_ckpt):
        print(f"Error: Diffusion Policy checkpoint not found: {args.diffusion_policy_ckpt}")
        sys.exit(1)
        
    test_dimensions(args.action_predictor_ckpt, args.diffusion_policy_ckpt, args.device)
