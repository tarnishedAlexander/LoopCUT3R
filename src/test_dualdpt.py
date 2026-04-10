import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dust3r.model import ARCroco3DStereoConfig, ARCroco3DStereo

def test_dualdpt():
    print("Testing ARCroco3DStereo with DualDPT...")
    config = ARCroco3DStereoConfig(
        img_size=(224, 224),
        patch_size=16,
        enc_embed_dim=768,
        enc_depth=24,         
        enc_num_heads=12,
        dec_embed_dim=512,
        dec_depth=8,
        dec_num_heads=16,
        head_type="dualdpt",
        pose_head=True,
        local_mem_size=100,
        freeze="none",
        pos_embed="RoPE100",
        output_mode="pts3d", # Ignored by dualdpt mapping but normally passed
    )
    
    config.num_blocks = 1     
    config.num_steps = 8      
    config.dec_num_steps_img = 8
    
    model = ARCroco3DStereo(config)
    print("Model instantiated successfully. Running dummy forward pass...")
    
    dummy_imgs = [torch.randn(1, 3, 224, 224)] # 1 view
    
    views = [
        {
            'img': img,
            'true_shape': torch.tensor([[224, 224]]),
            'img_mask': torch.tensor([True]),
            'ray_map': torch.zeros((1, 224, 224, 6)),
            'ray_mask': torch.tensor([False]),
            'camera_pose': torch.zeros((1, 7)),
            'reset': torch.tensor([True]),
            'idx': i
        } for i, img in enumerate(dummy_imgs)
    ]
    
    # We call standard forward to reach the downstream head
    out = model(views)
    
    print("Successfully produced output. Available keys:")
    print(out[0][0].keys())
    
    print("[Pass] Instantiation and Head binding successful.")

if __name__ == "__main__":
    test_dualdpt()
