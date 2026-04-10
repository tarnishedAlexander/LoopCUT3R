# CUT3R Complete Data Flow: From Raw Images to Encoder to Decoder

This is a detailed walkthrough of how data flows through CUT3R from preparation to encoder to decoder.

---

## 📊 Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CUT3R FULL DATA PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 1: DATA PREPARATION & LOADING
═════════════════════════════════════════════════════════════════════════════

  Raw Datasets (15 datasets)
  ├─ CO3D, ScanNet++, MegaDepth, DL3DV, etc.
  └─ Each has: images + camera poses + depth (optional)
       ↓
  Dataset Preprocessing
  (datasets_preprocess/*.py)
  ├─ Download & extract frames
  ├─ Calibrate camera intrinsics
  ├─ Resize to standard resolution (224 or 512)
  └─ Create train/val splits
       ↓
  Dataset Loaders
  (dust3r/datasets/base/base_multiview_dataset.py)
  ├─ Load multiple views per scene
  ├─ Apply augmentations (color jitter, cropping)
  ├─ Sample correspondence points
  └─ Return batch of views

STAGE 2: IMAGE PREPROCESSING
═════════════════════════════════════════════════════════════════════════════

  Input Batch: (num_views, batch_size, 3, H, W)
  Example: (4 views, 16 images per batch, 3 channels, 512×384 pixels)
       ↓
  Image Normalization
  ├─ Normalize to [-1, 1] or [0, 1]
  ├─ Standardize using ImageNet statistics
  └─ Convert PIL Images → PyTorch tensors
       ↓
  True Shape Tracking
  ├─ Store original image dimensions
  ├─ Handle aspect ratio variations
  ├─ Used by ManyAR_PatchEmbed for variable resolutions
  └─ Shape: (batch_size, 2) where 2 = [height, width]

STAGE 3: PATCH EMBEDDING (TOKENIZATION)
═════════════════════════════════════════════════════════════════════════════

  ┌─ Input: RGB Images (B, 3, H, W)
  │ Example: (16, 3, 512, 384)
  │
  ├─ Patch Embedding: PatchEmbedDust3R or ManyAR_PatchEmbed
  │
  ├─→ Step 1: Divide into patches
  │   ├─ Patch size: 16×16 pixels (PATCH_SIZE=16)
  │   ├─ Number of patches: (512/16) × (384/16) = 32 × 24 = 768 patches
  │   └─ Each patch: 16×16×3 = 768 values
  │
  ├─→ Step 2: Project patches to embedding dimension
  │   ├─ Conv2d(in_channels=3, out_channels=1024, kernel_size=16, stride=16)
  │   ├─ Output shape: (B, 1024, 32, 24)
  │   └─ Reshape to (B, num_patches, embedding_dim)
  │       Result: (16, 768, 1024)
  │
  ├─→ Step 3: Add positional embeddings (RoPE)
  │   ├─ Rotary Position Encoding (RoPE)
  │   ├─ Encodes 2D patch grid coordinates
  │   ├─ Position shape: (B, num_patches, 2)
  │   └─ Used during transformer layers (NOT added to tokens)
  │
  ├─→ Step 4: LayerNorm
  │   ├─ Normalize embeddings
  │   └─ Final output: (16, 768, 1024)
  │
  └─ Output: Patch tokens + Position embeddings

STAGE 4: ENCODER (CROCO BACKBONE)
═════════════════════════════════════════════════════════════════════════════

  ┌─ Input: Patch Tokens (B, num_patches, enc_embed_dim)
  │ Example: (16, 768, 1024)
  │
  ├─ FROZEN Encoder Processing
  │ (self.enc_blocks - list of 24 transformer blocks)
  │
  ├─→ For each of 24 transformer blocks:
  │   │
  │   ├─ Attention Block
  │   │ ├─ Query, Key, Value projections
  │   │ ├─ Multi-head attention (16 heads)
  │   │ ├─ RoPE positional encoding applied HERE
  │   │ │  └─ Rotates Q and K by position angle
  │   │ ├─ Scaled dot-product attention
  │   │ │  └─ sim(Q, K) / sqrt(d_k)
  │   │ └─ Output fusion
  │   │
  │   ├─ MLP (Feed-Forward Network)
  │   │ ├─ Linear(1024 → 4096)
  │   │ ├─ GELU activation
  │   │ ├─ Linear(4096 → 1024)
  │   │ └─ Dropout
  │   │
  │   └─ Residual connections + LayerNorm
  │
  ├─ After 24 blocks: Shape remains (16, 768, 1024)
  │
  ├─ Final LayerNorm
  │ └─ self.enc_norm: normalize final features
  │
  └─ Output: Encoded Features (16, 768, 1024)
     These are rich semantic + positional features

STAGE 5: RAY MAP ENCODING (Optional, for Multi-view)
═════════════════════════════════════════════════════════════════════════════

  ┌─ Input: Ray Maps (B, 6, H, W)
  │ └─ 6 channels: [ray_origin_x, ray_origin_y, ray_origin_z,
  │                 ray_direction_x, ray_direction_y, ray_direction_z]
  │
  ├─ Same process as image encoding:
  │ ├─ Patch embedding (6-channel version)
  │ ├─ 24 transformer blocks (separate enc_blocks_ray_map)
  │ └─ Final LayerNorm
  │
  └─ Output: Ray Map Features (16, 768, 1024)

STAGE 6: STATE INITIALIZATION
═════════════════════════════════════════════════════════════════════════════

  ┌─ Initialize persistent state tokens
  │
  ├─ State Size: 324 (default)
  │ └─ sqrt(324) = 18×18 grid for 2D positional encoding
  │
  ├─ Register Tokens (learnable parameters)
  │ ├─ self.register_tokens: nn.Embedding(state_size, enc_embed_dim)
  │ ├─ Creates 324 learnable vectors
  │ └─ Shape: (1, 324, 1024)
  │
  ├─ State Positional Encoding
  │ ├─ 2D grid: coordinates (i, j) for each state token
  │ ├─ Positions: (0,0), (0,1), ..., (17,17)
  │ └─ Shape: (B, 324, 2)
  │
  └─ Output: State features + positions
     (Used to initialize local memory in decoder)

═════════════════════════════════════════════════════════════════════════════

STAGE 7: MULTI-VIEW FUSION (_encode_views)
═════════════════════════════════════════════════════════════════════════════

  For multi-view sequences (4-64 views):
  
  ┌─ Stack all view encodings
  │ ├─ Image features: shape (num_views, batch_size, num_patches, enc_dim)
  │ ├─ Ray features: same shape
  │ └─ Positions: (num_views, batch_size, num_patches, 2)
  │
  ├─ Flatten batch: (num_views * batch_size, num_patches, enc_dim)
  │
  ├─ Combine image + ray features
  │ ├─ Where image_mask=True: use image features
  │ ├─ Where image_mask=False: use masked_img_token (learned placeholder)
  │ ├─ Where ray_mask=True: ADD ray features
  │ ├─ Where ray_mask=False: add masked_ray_map_token
  │ └─ Result: (num_views * batch_size, num_patches, enc_dim)
  │
  ├─ Chunk back to: (num_views, batch_size, num_patches, enc_dim)
  │
  └─ Output: Per-view encoded features ready for decoder

═════════════════════════════════════════════════════════════════════════════

KEY COMPONENTS SUMMARY
═════════════════════════════════════════════════════════════════════════════

1. ENCODER (FROZEN in training)
   Location: src/croco/models/croco.py (CroCoNet)
   └─ self.enc_blocks: 24 transformer layers
   └─ self.enc_norm: final normalization
   └─ Input: (B, num_patches, enc_embed_dim)
   └─ Output: (B, num_patches, enc_embed_dim)

2. PATCH EMBEDDING
   Location: src/dust3r/patch_embed.py
   └─ PatchEmbedDust3R: Fixed aspect ratio (224×224)
   └─ ManyAR_PatchEmbed: Variable aspect ratio (e.g., 512×384)
   └─ Both inherit from models.blocks.PatchEmbed

3. RAY MAP ENCODING
   Location: src/dust3r/model.py (_encode_ray_map)
   └─ self.enc_blocks_ray_map: Separate 24 blocks for ray features
   └─ self.patch_embed_ray_map: 6-channel patch embedding

4. STATE INITIALIZATION
   Location: src/dust3r/model.py (_encode_state)
   └─ self.register_tokens: learnable state embeddings
   └─ Creates persistent memory tokens
   └─ Initialized once at sequence start

5. MULTI-VIEW FUSION
   Location: src/dust3r/model.py (_encode_views)
   └─ Combines image + ray features from all views
   └─ Handles missing data with masking tokens
   └─ Outputs per-view feature sequences
```

---

## 🔍 Detailed Code Flow - Line by Line

### **1. ARCroco3DStereo.__init__() - Initialization**

```python
class ARCroco3DStereo(CroCoNet):  # Line 225
    def __init__(self, config: ARCroco3DStereoConfig):
        
        # Line 233: Initialize CroCo backbone
        super().__init__(croco_cfg)
        # This creates:
        # - self.enc_blocks (24 transformer layers)
        # - self.enc_norm (LayerNorm)
        # - self.patch_embed (CroCo's patch embedding)
        
        # Line 242-249: Create ray map encoding blocks
        self.enc_blocks_ray_map = nn.ModuleList([
            Block(...) for _ in range(24)  # Separate blocks for ray maps
        ])
        
        # Line 264: Set up decoder state management
        self._set_state_decoder(...)
        
        # Line 283: Set up downstream head (DPT, RGB, Pose)
        self.set_downstream_head(...)
```

### **2. Image Loading & Batching (from BaseMultiViewDataset)**

```python
# In training loop (src/train.py):
for batch in data_loader_train:
    # batch is a list of views
    # batch = [
    #     {'img': (B, 3, 512, 384),
    #      'true_shape': (B, 2),
    #      'camera_pose': (B, 4, 4),
    #      'img_mask': (B,),
    #      'ray_mask': (B,)},
    #     ...  (4 to 64 views)
    # ]
```

### **3. Forward Pass Entry Point**

```python
# Line 894: def forward(self, views, ret_state=False)
def forward(self, views, ret_state=False):
    # views = list of dicts, one per view in sequence
    # Each dict contains: {'img', 'ray_map', 'true_shape', 'img_mask', ...}
    
    return self._forward_impl(views, ret_state)
```

### **4. _forward_impl() - Main Forward Logic**

```python
# Line 920: def _forward_impl(self, views, ret_state=False)

# Step 1: Encode all views
shape, feat_ls, pos = self._encode_views(views)  # Line 921
# Output:
# - shape: (num_views, batch_size, 2)
# - feat_ls: list of features (last element has shape)
#   [(num_views, batch_size, num_patches, enc_dim)]
# - pos: (num_views, batch_size, num_patches, 2) - positions

# Step 2: Initialize state
feat = feat_ls[-1]  # (num_views, batch_size, num_patches, enc_dim)
state_feat, state_pos = self._init_state(feat[0], pos[0])
# state_feat: (batch_size, state_size, dec_embed_dim)
# state_pos: (batch_size, state_size, 2)

# Step 3: Initialize memory
mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
# mem: (batch_size, mem_size, 2*dec_embed_dim)

# Step 4: Iterate through each view in sequence
for i in range(len(views)):
    feat_i = feat[i]  # (batch_size, num_patches, enc_dim)
    pos_i = pos[i]    # (batch_size, num_patches, 2)
    
    # Forward through decoder with current state
    new_state_feat, dec = self._recurrent_rollout(
        state_feat, state_pos,
        feat_i, pos_i,
        pose_feat_i, pose_pos_i,
        init_state_feat
    )
    
    # Generate output for this view
    res = self._downstream_head(head_input, shape[i])
    ress.append(res)
    
    # Update state for next view
    state_feat = new_state_feat  # (batch_size, state_size, dec_dim)
    mem = new_mem                # (batch_size, mem_size, ...)
```

### **5. _encode_views() - Multi-View Encoding**

```python
# Line 659: def _encode_views(self, views, img_mask=None, ray_mask=None)

# Stack all views
imgs = torch.stack([view["img"] for view in views], dim=0)
# Shape: (num_views, batch_size, 3, H, W)

ray_maps = torch.stack([view["ray_map"] for view in views], dim=0)
# Shape: (num_views, batch_size, 6, H, W)

# Flatten for batch processing
imgs = imgs.view(-1, *imgs.shape[2:])
# Shape: (num_views * batch_size, 3, H, W)

# Process each view
selected_imgs = imgs[img_masks_flat]
img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
# img_out: (num_valid_imgs, num_patches, enc_dim)
# img_pos: (num_valid_imgs, num_patches, 2)

# Combine image + ray features
full_out[img_masks_flat] += img_out          # Add image features
full_out[ray_masks_flat] += ray_out          # Add ray features
full_out[~img_masks_flat] += self.masked_img_token    # Placeholder for missing
full_out[~ray_masks_flat] += self.masked_ray_map_token  # Placeholder

# Reshape back to per-view
shapes.chunk(len(views), dim=0)  # (num_views, batch_size, 2)
[out.chunk(len(views), dim=0) for out in full_out]  # Per-view features
```

### **6. _encode_image() - Image Encoding Pipeline**

```python
# Line 584: def _encode_image(self, image, true_shape)

# Step 1: Patch Embedding
x, pos = self.patch_embed(image, true_shape=true_shape)
# Input: (B, 3, H, W)
# x output: (B, num_patches, enc_embed_dim)
# pos output: (B, num_patches, 2)

# Step 2: Pass through 24 encoder blocks
for blk in self.enc_blocks:
    x = blk(x, pos)  # (B, num_patches, enc_embed_dim)
    # Each block:
    # - Multi-head attention with RoPE
    # - MLP
    # - Residual connections

# Step 3: Final normalization
x = self.enc_norm(x)
# x: (B, num_patches, enc_embed_dim)

return [x], pos, None
```

### **7. Transformer Block Structure**

```python
# Each enc_blocks[i] is a Block from dust3r/blocks.py

class Block:
    def forward(self, x, pos):
        # x: (B, N, D) - N patches, D dimension
        # pos: (B, N, 2) - patch positions
        
        # 1. Attention with RoPE
        x_att = attention(x, pos)  # RoPE applied in attention
        x = x + x_att              # Residual
        
        # 2. MLP
        x_mlp = mlp(x)
        x = x + x_mlp              # Residual
        
        return x  # (B, N, D)
```

### **8. Patch Embedding Details**

```python
# From patch_embed.py: PatchEmbedDust3R

def forward(self, x, true_shape):
    # Input: (B, 3, 512, 384)
    
    # 1. Convolutional projection
    x = self.proj(x)  # Conv2d(3, 1024, kernel_size=16, stride=16)
    # Output: (B, 1024, 32, 24)
    
    # 2. Flatten to patches
    x = x.flatten(2).transpose(1, 2)  # (B, 32*24, 1024) = (B, 768, 1024)
    
    # 3. LayerNorm
    x = self.norm(x)  # (B, 768, 1024)
    
    # 4. Get positional embeddings
    pos = self.position_getter(B, 32, 24, device)
    # pos: (B, 768, 2) - (x, y) coordinates
    
    return x, pos
```

---

## 🎯 Summary: Data Flow Path

```
Raw Images (512×384, 3 channels)
    ↓ [Patch Embedding: Conv 16×16 projection]
Patch Tokens (768 patches, 1024 dim)
    ↓ [Positional Encoding: RoPE coordinates]
Positioned Tokens
    ↓ [Encoder: 24 Transformer Blocks]
Encoded Features (768 patches, 1024 dim)
    ↓ [Multi-View Fusion: Stack all views]
Per-View Features (4-64 views, each with 768 patches, 1024 dim)
    ↓ [State Initialization: 324 state tokens]
State + Features Ready for Decoder
    ↓ [Decoder: 12 Transformer Blocks + LocalMemory]
Multi-Scale Decoder Output
    ↓ [Head: DPT/RGB/Pose]
Final Predictions (depth, confidence, RGB, poses)
```

---

## 📍 File Location Reference

| Component | File | Lines |
|-----------|------|-------|
| **ARCroco3DStereo class** | `src/dust3r/model.py` | 225-1124 |
| **Patch Embedding** | `src/dust3r/patch_embed.py` | 1-100+ |
| **Encoder blocks** | `src/croco/models/croco.py` | - |
| **Dataset loading** | `src/dust3r/datasets/base/base_multiview_dataset.py` | 1-547 |
| **Image utilities** | `src/dust3r/utils/image.py` | - |
| **Training loop** | `src/train.py` | 1-918 |
| **Forward impl** | `src/dust3r/model.py` | 920-1020 |
| **Encode views** | `src/dust3r/model.py` | 659-750 |
| **Encode image** | `src/dust3r/model.py` | 584-600 |

---

## ⚡ Key Insights

1. **Encoder is FROZEN** - The 24 transformer blocks in `self.enc_blocks` are pre-trained on CroCo and never updated during CUT3R training

2. **RoPE Positional Encoding** - Rotary Position Encoding is applied during attention, not added to tokens. This enables length generalization

3. **Dual Encoding Path** - Images and ray maps are encoded separately through `enc_blocks` and `enc_blocks_ray_map`, then combined

4. **State Tokens** - The 324 learnable state tokens are the core of CUT3R's persistent memory, initialized from image features

5. **Per-View Processing** - Each view goes through the same encoder path, then features are combined for multi-view fusion

6. **Masking Strategy** - Missing views use learned placeholder tokens (`masked_img_token`, `masked_ray_map_token`) instead of zeros

This is what makes CUT3R work: **pre-trained frozen encoders** producing rich features, then **state-aware decoding** that builds temporal consistency!
