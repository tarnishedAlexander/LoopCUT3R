"""The recurrent encoder: weight sharing, depth equivalence and TBPTL.

Runs on CPU with a small configuration, no checkpoint needed:

    python tests/test_recurrent_encoder.py
"""
import torch

import _paths  # noqa: F401
from croco.models.croco import CroCoNet, CrocoConfig


def small_model(num_blocks=2, num_steps=3, use_tbptl=False, tbptl_warmup=0):
    cfg = CrocoConfig(
        img_size=64, patch_size=16, enc_embed_dim=64, enc_depth=num_blocks * num_steps,
        enc_num_heads=4, dec_embed_dim=32, dec_depth=2, dec_num_heads=4,
    )
    cfg.num_blocks = num_blocks
    cfg.num_steps = num_steps
    cfg.use_tbptl = use_tbptl
    cfg.tbptl_warmup = tbptl_warmup
    return CroCoNet(cfg)


def test_blocks_are_shared_across_steps():
    """B blocks applied for S steps hold the parameters of B blocks, not B*S."""
    model = small_model(num_blocks=2, num_steps=3)
    assert len(model.enc_blocks) == 2
    assert model.num_steps == 3


def test_encoder_output_shape():
    torch.manual_seed(0)
    model = small_model().eval()
    img = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feat, pos, _ = model._encode_image(img)
    assert feat.shape == (2, 16, 64)
    assert pos.shape == (2, 16, 2)


def test_every_step_reuses_the_same_blocks():
    """return_all_blocks yields one feature map per step, all from the shared blocks."""
    torch.manual_seed(0)
    model = small_model(num_blocks=2, num_steps=3).eval()
    img = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        feats, _, _ = model._encode_image(img, return_all_blocks=True)
    assert len(feats) == 3


def test_tbptl_cuts_the_graph_before_the_warmup_steps():
    """With TBPTL the gradient stops at the warm-up steps and never reaches the image."""
    torch.manual_seed(0)
    model = small_model(use_tbptl=True, tbptl_warmup=2).train()
    img = torch.randn(2, 3, 64, 64, requires_grad=True)
    feat, _, _ = model._encode_image(img)
    feat.mean().backward()
    assert img.grad is None
    assert any(p.grad is not None for p in model.enc_blocks.parameters())


def test_without_tbptl_the_gradient_reaches_the_image():
    torch.manual_seed(0)
    model = small_model(use_tbptl=False).train()
    img = torch.randn(2, 3, 64, 64, requires_grad=True)
    feat, _, _ = model._encode_image(img)
    feat.mean().backward()
    assert img.grad is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
