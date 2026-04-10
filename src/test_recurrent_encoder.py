import sys
import os
import torch
import torch.nn as nn
import glob

# Add src and croco to python path
src_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'croco'))

from croco.models.croco import CroCoNet, CrocoConfig

def test_recurrent_croco():
    print("Testing Recurrent CroCoNet Initialization and Checkpoint Loading...")
    
    # Checkpoint path with glob matching
    ckpt_path = "/app/recurrentTransformer/raptor/src/runs/models/*final_weighted_False_autoregressive_False_distillation_True_teacher_False_mse_True_cosine_True_t_scale_False_swiglu_False_sigma_0.001_start_0_end_7_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_0_step_26940.pt"
    
    matched = glob.glob(ckpt_path)
    if not matched:
        print("Checkpoint not found!")
        return
    real_ckpt_path = matched[0]
    
    # Configure Recurrent CroCo options
    config = CrocoConfig(
        img_size=224,
        patch_size=16,
        enc_embed_dim=768,
        enc_depth=24,         
        enc_num_heads=12,
        dec_embed_dim=512,
        dec_depth=8,
        dec_num_heads=16,
    )
    
    # Adding recurrent params
    config.num_blocks = 1     
    config.num_steps = 8      # Evaluates 8 depth steps recursively
    config.recurrent_ckpt = real_ckpt_path
    
    print("Initializing Model...")
    model = CroCoNet(config)
    print(f"Model initialized successfully. Total params: {sum(p.numel() for p in model.parameters())}")
    
    # Perform a dummy forward pass
    print("Running forward pass...")
    dummy_img = torch.randn(2, 3, 224, 224)
    out, pos, mask = model._encode_image(dummy_img, do_mask=False)
    
    print(f"Output shape: {out.shape}")
    print("Success! Recurrent Encoder is functional.")

if __name__ == "__main__":
    test_recurrent_croco()
