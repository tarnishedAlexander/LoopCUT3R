import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'croco'))
from croco.models.croco import CroCoNet, CrocoConfig

def test_tbptl():
    print("--- Testing TBPTL Truncation ---")
    config = CrocoConfig(
        img_size=224, patch_size=16, enc_embed_dim=768, enc_depth=24,
        enc_num_heads=12, dec_embed_dim=512, dec_depth=8, dec_num_heads=16,
    )
    config.num_blocks = 1
    config.num_steps = 8
    
    # TEST 1: TBPTL enabled
    config.use_tbptl = True
    config.tbptl_warmup = 2
    model1 = CroCoNet(config)
    model1.train()
    img1 = torch.randn(2, 3, 224, 224, requires_grad=True)
    out1, _, _ = model1._encode_image(img1)
    loss1 = out1.mean()
    loss1.backward()
    
    if img1.grad is None:
        print("[Pass] TBPTL correctly severed the backward graph! The input image gradients are safely None.")
    else:
        print("[Fail] Error in TBPTL: Gradients leaked to the input image.")
        
    # TEST 2: TBPTL disabled
    print("--- Testing Standard Graph ---")
    config.use_tbptl = False
    model2 = CroCoNet(config)
    model2.train()
    img2 = torch.randn(2, 3, 224, 224, requires_grad=True)
    out2, _, _ = model2._encode_image(img2)
    loss2 = out2.mean()
    loss2.backward()
    
    if img2.grad is not None:
        print("[Pass] Standard Model correctly flows gradients all the way to the input image.")
    else:
        print("[Fail] Error in standard operation: Gradients did not reach the input image.")

if __name__ == "__main__":
    test_tbptl()
