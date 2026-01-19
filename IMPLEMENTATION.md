# PixelHDM Implementation Documentation

> **PixelHDM**: Pixel Home-scale Diffusion Model

**Version**: 1.2.0
**Date**: 2026-01-19

---

## Table of Contents

1. [Configuration Parameters](#1-configuration-parameters) (1.1-1.16)
2. [Model Architecture](#2-model-architecture) (2.1-2.5)
3. [Attention Mechanisms](#3-attention-mechanisms) (3.1-3.6)
4. [Loss Functions](#4-loss-functions) (4.1-4.4)
5. [Training System](#5-training-system) (5.1-5.10)
6. [Inference System](#6-inference-system) (6.1-6.5)
7. [Embedding Layers](#7-embedding-layers) (7.1-7.6)
8. [Initialization Strategies](#8-initialization-strategies) (8.1-8.6)

---

## 1. Configuration Parameters

### 1.1 Core Dimensions

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `hidden_dim` | 1024 | Matches Qwen3-0.6B hidden size, no projection needed |
| `pixel_dim` | 16 | Compact pixel representation, p² × D_pix = 4096 |
| `patch_size` | 16 | Matches DINOv3 ViT-B/16, no interpolation needed |
| `head_dim` | 64 | Standard for modern transformers (1024 / 16 heads) |

### 1.2 Layer Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `patch_layers` | 16 | Deep semantic processing, matches DiT-XL |
| `pixel_layers` | 4 | Lightweight pixel refinement (1:4 ratio) |
| `num_heads` | 16 | GQA with 16 Q heads |
| `num_kv_heads` | 4 | 4:1 GQA ratio, 75% KV cache reduction |

### 1.3 GQA 4:1 Ratio Justification

$$\text{KV Memory} = \frac{4}{16} = 25\% \text{ of MHA}$$

- **Why 4:1**: Balance between efficiency and expressiveness
- **Alternatives considered**:
  - 1:1 (MHA): No savings
  - 8:1: Too aggressive, quality degradation
  - 16:1 (MQA): Extreme savings but limited capacity

### 1.4 MLP Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `mlp_ratio` | 3.0 | SwiGLU uses 3× instead of GELU's 4× |
| `mlp_type` | "swiglu" | Better gradient flow, LLaMA/Qwen standard |

**SwiGLU vs GELU FFN**:
```
GELU FFN:   FFN(x) = GELU(xW₁)W₂
SwiGLU:     FFN(x) = (SiLU(xW_gate) ⊙ xW_up)W_down
```

### 1.5 Time Sampling (Flow Matching)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `time_p_mean` | 0.0 | SD3/PixelHDM convention |
| `time_p_std` | 1.0 | SD3/PixelHDM convention |
| `time_eps` | 0.05 | Avoid numerical instability at t→0 or t→1 |

**Logit-Normal Sampling**:
$$t = t_{eps} + (1 - 2 \cdot t_{eps}) \cdot \sigma(p_{mean} + p_{std} \cdot u), \quad u \sim \mathcal{N}(0, 1)$$

### 1.6 REPA Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `repa_lambda` | 0.5 | Half weight of V-Loss |
| `repa_early_stop` | 250000 | Disable after semantic structure established |
| `repa_align_layer` | 8 | Middle-to-late layer for semantic alignment |
| `repa_hidden_size` | 768 | DINOv3 ViT-B feature dimension |

### 1.7 Frequency Loss Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `freq_loss_quality` | 90 | High-quality JPEG quantization weights |
| `freq_loss_block_size` | 8 | Standard JPEG 8×8 DCT blocks |
| `freq_loss_use_ycbcr` | True | Perceptually-aligned color space |

### 1.8 Input/Output Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `in_channels` | 3 | RGB input |
| `out_channels` | 3 | RGB output (velocity prediction) |
| `max_resolution` | 1024 | Maximum supported resolution |
| `time_convention` | "pixelhdm" | t=0 noise, t=1 clean (opposite to standard) |
| `prediction_type` | "x" | Predict clean image directly |

### 1.9 Regularization Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `dropout` | 0.0 | No dropout (diffusion uses noise as regularization) |
| `attention_dropout` | 0.0 | No attention dropout |
| `cfg_dropout` | 0.1 | 10% unconditional training for CFG |

### 1.10 Optimization Flags

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `use_flash_attention` | True | 3-8× attention speedup, O(N) memory |
| `use_gradient_checkpointing` | True | 50% memory reduction |
| `zero_init_output` | True | Initial predictions ≈ 0 for stable training |

### 1.11 Normalization Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `norm_type` | "rmsnorm" | Faster than LayerNorm, LLaMA/Qwen standard |
| `norm_eps` | 1e-6 | Standard epsilon for RMSNorm |

### 1.12 Gated Attention Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `gate_type` | "headwise" | Per-head gating (16K params vs 1M for elementwise) |
| `gate_activation` | "sigmoid" | Bounded [0,1] output |
| `gate_bias` | False | Zero-init weights → initial gate = 0.5 |

### 1.13 AdaLN Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `adaln_num_params` | 6 | gamma1, beta1, alpha1, gamma2, beta2, alpha2 |
| `adaln_init_gain` | 0.0001 | init |

### 1.14 mRoPE Extended Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `mrope_theta` | 10000.0 | LLaMA/Qwen standard frequency base |
| `mrope_text_max_len` | 512 | Match text_max_length |
| `mrope_img_max_len` | 65536 | Support up to 256×256 patches |

### 1.15 Embedding Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `time_embed_dim` | 256 | Sinusoidal embedding dimension (before MLP) |
| `bottleneck_dim` | 256 | PatchEmbedding intermediate dimension |
| `max_patches` | 4096 | Maximum patch count (64×64 grid) |

### 1.16 Inference Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `default_num_steps` | 50 | Balance between quality and speed |
| `default_guidance_scale` | 7.5 | Standard CFG scale |
| `default_sampler_method` | "heun" | 2nd order, good quality/speed tradeoff |

---

## 2. Model Architecture

### 2.1 Dual-Path Design

```
Input: Noised Image x_t (B, H, W, 3)
        │
        ├────────────────────────────────────────┐
        │                                        │
        ▼                                        ▼
┌─────────────────┐                    ┌─────────────────┐
│ 16×16 Patchify  │                    │  1×1 Patchify   │
│ (PatchEmbedding)│                    │ (PixelEmbedding)│
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         ▼                                      │
┌─────────────────┐                             │
│   DiT Blocks    │                             │
│  (16 layers)    │─── REPA features ──►        │
└────────┬────────┘                             │
         │                                      │
         ▼                                      │
┌─────────────────┐                             │
│    s_cond =     │                             │
│ semantic + t +  │                             │
│ pooled_text     │                             │
└────────┬────────┘                             │
         │     ┌────────────────────────────────┘
         │     │
         ▼     ▼
    ┌─────────────────┐
    │   PiT Blocks    │
    │  (4 layers)     │
    │ + TokenCompact  │
    └────────┬────────┘
             │
             ▼
        Output: v (velocity)
```

### 2.2 s_cond Conditioning Mechanism

The condition signal `s_cond` fuses three information sources:

```python
s_cond = semantic_tokens + t_embed.unsqueeze(1) + pooled_text_embed.unsqueeze(1)
```

| Component | Shape | Source | Purpose |
|-----------|-------|--------|---------|
| `semantic_tokens` | (B, L, D) | Patch DiT output | Spatial semantic info |
| `t_embed` | (B, 1, D) | TimeEmbedding | Timestep conditioning |
| `pooled_text_embed` | (B, 1, D) | Text encoder | Global text semantics |

### 2.3 Sandwich Norm Design

```
x ────────────────────────────────────────► (+) ─► output
    │                                         ▲
    │                                         │ × alpha
    ▼                                         │
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│Pre-Norm │→γ*│ +β      │→│   Op    │→│Post-Norm│
│(RMSNorm)│   │(AdaLN)  │  │(Attn/FF)│  │(RMSNorm)│
└─────────┘   └─────────┘   └─────────┘   └─────────┘
```

**Why Sandwich Norm**: Maximum training stability with smooth gradient flow.

### 2.4 Gradient Checkpointing

- **Granularity**: Block-level (not per-operation)
- **Implementation**: `torch.utils.checkpoint` with `use_reentrant=False`
- **Nested**: Inner attention uses `use_checkpoint=False` (outer block handles it)

### 2.5 PixelHDMForT2I Wrapper

**Purpose**: Complete text-to-image model with all components integrated.

```python
class PixelHDMForT2I(nn.Module):
    def __init__(self, config, text_encoder, dino_encoder):
        self.pixelhdm = PixelHDM(config)      # Core diffusion model
        self.text_encoder = text_encoder       # Qwen3-0.6B (frozen)
        self.dino_encoder = dino_encoder       # DINOv3 ViT-B (frozen, REPA)
```

**Forward Flow**:
```
prompt → text_encoder → (text_embed, pooled_embed)
                              ↓
x_t, t ─────────────────► pixelhdm ────► v_pred
                              ↓
x_clean → dino_encoder → dino_features (for REPA loss)
```

**Component Initialization**:
- `load_text_encoder=True`: Load Qwen3-0.6B from HuggingFace
- `load_dino_encoder=True`: Load DINOv3 from local path
- Both encoders frozen during training

---

## 3. Attention Mechanisms

### 3.1 GQA Implementation

**KV Head Expansion**:
```python
def repeat_kv(k, v, n_rep=4):
    # (B, 4, L, 64) → (B, 16, L, 64)
    k = k.unsqueeze(2).expand(B, 4, 4, L, 64)
    k = k.reshape(B, 16, L, 64)
    return k, v
```

### 3.2 Gated Attention

$$\mathbf{Y}' = \mathbf{Y} \odot \sigma(\mathbf{X}\mathbf{W}_g)$$

| Mode | Output Shape | Parameters |
|------|--------------|------------|
| `headwise` (default) | (B, 16, L, 1) | 16K |
| `elementwise` | (B, 16, L, 64) | 1M |

**Initialization**: Zero weights → initial gate = 0.5 → 50% passthrough

### 3.3 Token Compaction

**Complexity Reduction**: O(L² × p⁴) → O(L²)

```
Input: (B, L, 256, 16)
   ↓ Compress: Linear(4096 → 1024)
(B, L, 1024)
   ↓ RMSNorm
   ↓ GQA Attention
   ↓ RMSNorm
   ↓ Expand: Linear(1024 → 4096), gain=0.1
(B, L, 256, 16)
   ↓ + Residual
Output: (B, L, 256, 16)
```

**expand_gain=0.1**: adaLN-Zero style, initial output ≈ identity mapping.

### 3.4 mRoPE (Multi-axis RoPE)

**Dimension Allocation** (head_dim=64):
- Text: 16 dims (25%)
- Image Height: 24 dims (37.5%)
- Image Width: 24 dims (37.5%)

**Frequency Base**: θ = 10000 (LLaMA/Qwen standard)

### 3.5 QK Normalization

**Purpose**: Stabilize attention logits, prevent exploding dot products.

```python
# Applied before attention computation
q = F.normalize(q, dim=-1)  # L2 normalize over head_dim
k = F.normalize(k, dim=-1)  # L2 normalize over head_dim

# Then scale by learned temperature
attn = (q @ k.T) * temperature  # temperature ≈ √head_dim
```

**Benefits**:
- Bounded attention logits ∈ [-1, 1] × temperature
- Prevents NaN during bf16 training
- More stable gradient flow

**When Used**: Automatically applied in `GatedMultiHeadAttention` when `qk_norm=True`.

### 3.6 Flash Attention Configuration

**Conditions for Flash Attention**:
```python
use_flash = (
    use_flash_attention and              # Config flag
    torch.cuda.is_available() and        # GPU present
    hasattr(F, 'scaled_dot_product_attention')  # PyTorch 2.0+
)
```

**SDPA Parameters**:
```python
F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,           # No explicit mask (handled internally)
    dropout_p=0.0,            # No dropout
    is_causal=False,          # NOT causal (bidirectional image attention)
    scale=None,               # Auto-compute 1/√head_dim
)
```

**Critical**: `is_causal=False` - Image tokens attend bidirectionally, unlike LLM.

**Fallback**: Standard attention computation when Flash unavailable.

---

## 4. Loss Functions

### 4.1 Triple Loss System

$$L = L_v + \lambda_{freq} \times L_{freq} + \lambda_{repa} \times L_{REPA}$$

### 4.2 V-Loss (Velocity Prediction)

$$L_v = \mathbb{E}_{x, \varepsilon, t}\left[\|v_{pred} - (x - \varepsilon)\|^2\right]$$

**Why V-Prediction**:
- Avoids 1/(1-t) error amplification near t=1
- Direct ODE correspondence: dz/dt = v
- More stable for pixel-level predictions

### 4.3 Frequency Loss

$$L_{freq} = \frac{1}{N} \sum_{i,j} W_{i,j} \cdot (V^{pred}_{DCT} - V^{target}_{DCT})^2$$

**Components**:
- DCT-II transform on 8×8 blocks
- JPEG quantization weights (Q=90)
- YCbCr color space conversion

### 4.4 REPA Loss

$$L_{REPA} = 1 - \frac{1}{BL} \sum_{b,l} \cos(h_{b,l}, y_{b,l})$$

**iREPA Improvements** (arXiv:2512.10794):
- Conv2d projection (replaces MLP)
- Spatial normalization

**Early Stop**: Disabled after 250K steps to avoid conflicting with denoising objective.

---

## 5. Training System

### 5.1 Flow Matching Time Convention

| t | State |
|---|-------|
| 0 | Pure noise |
| 1 | Clean image |

**Interpolation**: z_t = t × x + (1 - t) × ε

### 5.2 Learning Rate Schedule

**SteppedCosineRestart**: Peak AND valley decay together

```python
cycle_peak = max(global_min_lr, base_lr × decay_rate^cycle)
cycle_min = max(global_min_lr, cycle_min_lr × decay_rate^cycle)
lr = cycle_min + (cycle_peak - cycle_min) × 0.5 × (1 + cos(π × progress))
```

**Default Schedule** (16 epochs, restart_epochs=1):
```
Epoch 0:  1.0e-4 → 5.0e-5
Epoch 1:  9.0e-5 → 4.5e-5  (×0.9)
Epoch 15: 2.1e-5 → 1.0e-5
```

### 5.3 EMA

| Parameter | Value | Effective Window |
|-----------|-------|------------------|
| decay | 0.999 | ~1000 steps |

### 5.4 Bucket System

**AspectRatioBucket**:
- Aspect-ratio priority matching (0.7 weight)
- No upscaling allowed
- Step size: patch_size × 4 = 64

**BufferedShuffleSampler**:
- Chunks of similar-sized buckets
- Shuffle chunks, not individual samples
- Stable GPU memory usage

### 5.5 OOM Recovery

```
Retry 1: Clear CUDA cache + GC
Retry 2: Halve batch size
Retry 3: Halve again
Retry 4+: Skip batch
```

### 5.6 Optimizer Configuration

**AdamW Parameters**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `betas` | (0.9, 0.999) | Standard AdamW defaults |
| `eps` | 1e-8 | Numerical stability |
| `weight_decay` | 0.01 | Mild L2 regularization |

**Gradient Accumulation**:
```python
effective_batch_size = batch_size × gradient_accumulation_steps
optimizer.step()  # Only after accumulation_steps forward passes
```

### 5.7 ZClip Gradient Clipping

**Adaptive Threshold Mechanism**:
```python
# Track gradient norm EMA
grad_norm_ema = ema_decay * grad_norm_ema + (1 - ema_decay) * current_grad_norm

# Clip threshold adapts to training dynamics
threshold = clip_factor * grad_norm_ema  # Default: clip_factor=2.5
```

**Default Configuration**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `threshold` | 2.5 | Clip at 2.5× EMA norm |
| `ema_decay` | 0.99 | Smooth norm tracking |

**Advantage over Fixed Clipping**: Adapts to different training phases (high initial gradients → low gradients).

### 5.8 Mixed Precision Training

**Strategy**: bf16 with selective fp32 operations.

```python
# Model in bf16
model = model.to(torch.bfloat16)

# Critical operations stay fp32
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(inputs)
    loss = loss_fn(output, targets)  # Loss computed in fp32

# Gradient scaling NOT needed for bf16 (unlike fp16)
loss.backward()
optimizer.step()
```

**Why bf16 over fp16**:
- Dynamic range: bf16 ≈ fp32 (8-bit exponent)
- No loss scaling needed
- Native hardware support on modern GPUs

### 5.9 CFG Dropout Mechanism

**Purpose**: Enable classifier-free guidance during inference.

**Training**:
```python
# With probability cfg_dropout (10%), use unconditional embedding
if random.random() < cfg_dropout:
    text_embed = null_text_embed  # Zero or learned null embedding
```

**Inference (CFG)**:
```python
v_cond = model(x_t, t, text_embed)    # Conditional prediction
v_uncond = model(x_t, t, null_embed)  # Unconditional prediction
v_cfg = v_uncond + scale * (v_cond - v_uncond)  # Guided output
```

**cfg_dropout=0.1 Rationale**: 10% unconditional rate provides enough signal for CFG without hurting conditional quality.

### 5.10 CPU Memory Checkpoint

**For Extremely Large Models**:
```python
# Offload activations to CPU during forward pass
checkpoint_fn = checkpoint_sequential_cpu if cpu_offload else checkpoint
```

**Trade-off**: 10-20× slower but enables training models that don't fit in GPU.

---

## 6. Inference System

### 6.1 Samplers

| Sampler | NFE (50 steps) | Order | Use Case |
|---------|----------------|-------|----------|
| Euler | 50 | 1st | Fast preview |
| **Heun** | 99 | 2nd | **Default** |
| DPM++ | 50 | Multi-step | Low-step quality |
| DPM++ 2S | 50 | Multi-step | Second-order DPM solver |

**DPM++ 2S (Second-Order Ancestral)**:
```python
# Two-step predictor-corrector method
h = t_{i+1} - t_i
s = t_i + 0.5 * h  # Midpoint

# Predictor: Euler step to midpoint
u = z_t + h/2 * v(z_t, t_i)

# Corrector: Full step using midpoint derivative
z_{i+1} = z_t + h * v(u, s)
```

**When to Use Each**:
- **Euler**: Development/debugging (fastest)
- **Heun**: Production (quality/speed balance)
- **DPM++**: Low step count (< 30 steps)
- **DPM++ 2S**: Best quality at moderate steps

### 6.2 CFG Implementation

**Standard CFG**:
$$\hat{v} = v_{uncond} + s \times (v_{cond} - v_{uncond})$$

**Rescaled CFG** (prevents oversaturation):
$$\hat{v}_{final} = \phi \times (v_{cfg} \times \frac{\sigma_{cond}}{\sigma_{cfg}}) + (1 - \phi) \times v_{cfg}$$

### 6.3 Timesteps

$$t_i = t_{eps} + \frac{i}{N} \times (1 - 2 \times t_{eps}), \quad i \in \{0, ..., N\}$$

**Why [t_eps, 1-t_eps] instead of [0, 1]**:
- t=0: λ(0) = log(0/1) = -∞
- t=1: λ(1) = log(1/0) = +∞
- Numerical stability with t_eps=0.05

### 6.4 CFG Scheduling Strategies

**Available Schedule Types**:

| Schedule | Formula | Use Case |
|----------|---------|----------|
| `constant` | scale(t) = s | Standard CFG |
| `linear` | scale(t) = s × (1 - t) | Reduce CFG as approaching clean |
| `cosine` | scale(t) = s × 0.5 × (1 + cos(πt)) | Smooth decay |
| `quadratic` | scale(t) = s × (1 - t)² | Aggressive late-stage reduction |

**CFGWithInterval**:
```python
# Only apply CFG during specific timestep range
if t_start < t < t_end:
    v = apply_cfg(v_cond, v_uncond, scale)
else:
    v = v_cond  # No CFG outside interval
```

**Default**: `t_start=0.0, t_end=1.0` (full CFG throughout).

**When to Use Interval**: Reduce artifacts at extreme timesteps.

### 6.5 Pipeline Optimizations

**torch.compile Support**:
```python
pipeline = PixelHDMPipeline(model)
pipeline.compile_model()  # Optional: JIT compile for 10-30% speedup
```

**Batch Generation**:
```python
output = pipeline(
    prompt="sunset",
    num_images_per_prompt=4,  # Generate 4 images per prompt
)
```

**Deterministic Generation**:
```python
output = pipeline(
    prompt="sunset",
    seed=42,  # Fixed seed for reproducibility
)
```

---

## 7. Embedding Layers

### 7.1 PatchEmbedding (16×16)

**Bottleneck Design**: 768 → 256 → 1024

```
unfold(16×16) → Linear(768→256) → SiLU → Linear(256→1024)
```

**Parameter Savings**: 42% fewer params than direct projection

### 7.2 PixelEmbedding (1×1)

**Purpose**: Preserve high-frequency details

```
Linear(3→16) per-pixel → reshape to (B, L, 256, 16)
```

### 7.3 TimeEmbedding

**Sinusoidal + MLP**:
$$embed(t) = MLP([\sin(1000t \cdot f_0), ..., \cos(1000t \cdot f_{127})])$$

**MLP Structure**: Linear(256→1024) → SiLU → Linear(1024→1024)

### 7.4 Text Encoder

| Property | Value | Rationale |
|----------|-------|-----------|
| Model | Qwen3-0.6B | hidden_size=1024 matches |
| Frozen | True | Stability, prevent forgetting |
| Pooling | Last token | Causal LM accumulates info |

### 7.5 PixelPatchify (Output Projection)

**Purpose**: Convert pixel features back to RGB image.

```
Input: (B, L, 256, 16)
    ↓ Reshape: (B, L×256, 16) = (B, H×W, 16)
    ↓ Linear(16 → 3): Xavier uniform initialization
Output: (B, H, W, 3)
```

**Critical Initialization**:
```python
# CORRECT: Xavier for proper output variance
nn.init.xavier_uniform_(self.proj.weight)
nn.init.zeros_(self.proj.bias)

# WRONG: Small std causes V-Loss to plateau at ~0.73
# nn.init.trunc_normal_(self.proj.weight, std=0.02)  # Don't do this!
```

**Why Xavier**:
- Output std ≈ 1.30 (matches v_target std ≈ 1.15)
- Small std=0.02 → output std = 0.08 (only 7% coverage)

### 7.6 LearnedPositionalEmbedding

**Alternative to mRoPE** (not currently used):
```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim):
        self.embedding = nn.Embedding(max_len, hidden_dim)

    def forward(self, seq_len):
        positions = torch.arange(seq_len)
        return self.embedding(positions)
```

**When to Use**: For sequence-agnostic models that don't need 2D structure awareness.

**Current Choice**: mRoPE preferred for explicit 2D image structure encoding.

---

## 8. Initialization Strategies

### 8.1 Basic Initialization

| Layer | Weight | Bias |
|-------|--------|------|
| `nn.Linear` | Xavier Uniform | zeros |
| `nn.Embedding` | Normal(std=0.02) | N/A |

### 8.2 AdaLN Initialization (CRITICAL)

**TokenAdaLN**:
```
gamma1=1, beta1=0, alpha1=1
gamma2=1, beta2=0, alpha2=1
```

**PixelwiseAdaLN**:
```
gamma1=init_gain, beta1=0, alpha1=1
gamma2=init_gain, beta2=0, alpha2=1
```
預設 `init_gain` 由 `config.adaln_init_gain` 設定。

**重要 (2026-01-19)**: `cond_norm` (RMSNorm) 已從 PixelwiseAdaLN 中**移除**。
它原本是為了「恢復信號強度」而添加，但實際上破壞了 99.7% 的文字條件信號。
- 問題: `cond_expand` 將不同的 s_cond 投影到幾乎平行的向量 (cosine_sim=0.998)
- RMSNorm 只保留方向 → 不同文字輸入在歸一化後變得相同
- 信號保留率: 有 cond_norm=0.3%, 無 cond_norm=54%

### 8.3 _reinit_adaln() Necessity

**Problem**: `apply(_basic_init)` overwrites all Linear.bias to zero
**Solution**: Re-initialize AdaLN after basic init

```python
def _init_weights(self):
    self.apply(_basic_init)
    self._init_output_proj()
    self._reinit_adaln()  # CRITICAL: restore AdaLN bias
```

### 8.4 TokenCompaction expand_gain

```python
nn.init.xavier_uniform_(self.expand.weight, gain=0.1)
```

**Effect**: Initial Token Compaction ≈ identity mapping

### 8.5 RMSNorm vs LayerNorm

| Property | RMSNorm | LayerNorm |
|----------|---------|-----------|
| Computation | 1× mean | 2× mean + variance |
| Parameters | γ only | γ + β |
| Adopted by | LLaMA, Qwen | GPT, BERT |

**eps=1e-6**: Standard for RMSNorm (vs 1e-5 for LayerNorm)

### 8.6 DINOv3 Encoder Optimizations

**SDPA (Scaled Dot-Product Attention)**:
```python
# Replace standard attention with SDPA for 2-3× speedup
F.scaled_dot_product_attention(q, k, v)  # Auto-selects best backend
```

**Inference Mode**:
```python
with torch.inference_mode():  # Faster than no_grad()
    features = dino_encoder(images)
```

**torch.compile**:
```python
# JIT compile for additional 10-30% speedup
dino_encoder = torch.compile(dino_encoder, mode='reduce-overhead')
```

**LRU Cache for Features**:
```python
@lru_cache(maxsize=1000)
def get_dino_features(image_hash):
    return dino_encoder(image)  # Cache repeated images
```

**Why DINOv3 Only** (No DINOv2 Fallback):
- DINOv3: patch_size=16 → perfect match with PixelHDM
- DINOv2: patch_size=14 → requires interpolation, quality loss

---

## Appendix: File Locations

| Component | Path |
|-----------|------|
| Config | `src/config/pixelhdm_config.py` |
| Core Model | `src/models/pixelhdm/core.py` |
| T2I Model | `src/models/pixelhdm/t2i.py` |
| PatchBlock | `src/models/blocks/patch/core.py` |
| PixelBlock | `src/models/blocks/pixel/core.py` |
| TokenCompaction | `src/models/attention/token_compaction.py` |
| GatedAttention | `src/models/attention/gated_attention.py` |
| mRoPE | `src/models/layers/rope/mrope.py` |
| TokenAdaLN | `src/models/layers/adaln/token_adaln.py` |
| PixelwiseAdaLN | `src/models/layers/adaln/pixelwise_adaln.py` |
| PatchEmbedding | `src/models/layers/embedding/patch.py` |
| PixelEmbedding | `src/models/layers/embedding/pixel.py` |
| PixelPatchify | `src/models/layers/embedding/pixel_patchify.py` |
| TimeEmbedding | `src/models/layers/embedding/time.py` |
| Text Encoder | `src/models/encoders/text_encoder.py` |
| DINOv3 Encoder | `src/models/encoders/dinov3.py` |
| V-Loss | `src/training/losses/vloss.py` |
| Freq Loss | `src/training/losses/freq/` |
| REPA Loss | `src/training/losses/repa/` |
| Combined Loss | `src/training/losses/combined.py` |
| Flow Matching | `src/training/flow_matching/` |
| Trainer | `src/training/trainer/core.py` |
| Scheduler | `src/training/optimization/scheduler/` |
| ZClip | `src/training/optimization/zclip.py` |
| EMA | `src/training/optimization/ema.py` |
| Bucket System | `src/training/bucket/` |
| Pipeline | `src/inference/pipeline/` |
| Samplers | `src/inference/sampler/` |
| CFG | `src/inference/cfg/` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2.0 | 2026-01-19 | **CRITICAL**: 從 PixelwiseAdaLN 移除 cond_norm (它破壞了 99.7% 文字信號) |
| 1.1.0 | 2026-01-08 | Added missing config params, QK norm, Flash Attention, optimizer, ZClip, CFG scheduling |
| 1.0.0 | 2026-01-08 | Initial version |

---

*最後更新: 2026-01-19*
