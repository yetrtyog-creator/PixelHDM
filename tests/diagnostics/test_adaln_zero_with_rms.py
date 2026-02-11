"""
AdaLN-Zero with RMS Matching - Combining Both Approaches

Goal: Test if AdaLN-Zero + RMS matching provides the best of both worlds:
- AdaLN's LN normalization (acts on unit scale)
- RMS matching's relative energy control (delta/x stays constant)

Based on: GPT/architecture_convergence_improvement/AdaLN_Zero_math_analysis.md
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class BroadcastBiasSync(nn.Module):
    """Current broadcast bias form (baseline)"""
    def __init__(self, hidden_dim: int, rho: float = 0.10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rho = rho
        self.eps = 1e-8

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_param = nn.Parameter(torch.tensor(0.0))

        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def _rms(self, t: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
        return (t.pow(2).mean(dim=dim, keepdim=keepdim) + self.eps).sqrt()

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        bias = self.proj(s)
        x_rms = self._rms(x, dim=-1, keepdim=False).mean(dim=-1, keepdim=True)
        b_rms = self._rms(bias, dim=-1, keepdim=True)
        bias_hat = bias * (self.rho * x_rms / (b_rms + self.eps))
        gate = torch.sigmoid(self.gate_param)
        delta = gate * bias_hat
        return x + delta.unsqueeze(1)


class AdaLNZeroSync(nn.Module):
    """AdaLN-Zero without RMS matching (original)"""
    def __init__(self, hidden_dim: int, zero_init: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.gate_param = nn.Parameter(torch.tensor(0.0))

        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        gb = self.mlp(s)
        gamma, beta = gb.chunk(2, dim=-1)
        g = torch.sigmoid(self.gate_param)

        # AdaLN-Zero: y * (1 + g*gamma) + g*beta
        y_scaled = y * (1 + g * gamma.unsqueeze(1)) + g * beta.unsqueeze(1)
        delta = y_scaled - y  # = g * (y * gamma + beta)

        return x + delta


class AdaLNZeroWithRMS(nn.Module):
    """AdaLN-Zero WITH RMS matching - combining both approaches"""
    def __init__(self, hidden_dim: int, rho: float = 0.10, zero_init: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rho = rho
        self.eps = 1e-8

        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.gate_param = nn.Parameter(torch.tensor(0.0))

        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
            nn.init.zeros_(self.mlp[-1].bias)

    def _rms(self, t: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
        return (t.pow(2).mean(dim=dim, keepdim=keepdim) + self.eps).sqrt()

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # LayerNorm
        y = self.ln(x)  # (B, N, D), unit scale

        # Get gamma, beta from condition
        gb = self.mlp(s)  # (B, 2D)
        gamma, beta = gb.chunk(2, dim=-1)  # each (B, D)

        # Gate
        g = torch.sigmoid(self.gate_param)

        # Compute raw delta from AdaLN
        # delta_raw = g * (y * gamma + beta)
        y_scaled = y * (1 + g * gamma.unsqueeze(1)) + g * beta.unsqueeze(1)
        delta_raw = y_scaled - y  # (B, N, D)

        # RMS matching: scale delta to be rho fraction of x
        x_rms = self._rms(x, dim=-1, keepdim=True)  # (B, N, 1)
        delta_rms = self._rms(delta_raw, dim=-1, keepdim=True)  # (B, N, 1)

        # Scale delta to match target RMS
        delta_hat = delta_raw * (self.rho * x_rms / (delta_rms + self.eps))

        return x + delta_hat


def test_all_three_methods():
    """Compare all three methods"""
    print("=" * 80)
    print("Comparing: Broadcast vs AdaLN-Zero vs AdaLN-Zero+RMS")
    print("=" * 80)

    D = 256
    B, N = 4, 64

    broadcast = BroadcastBiasSync(D, rho=0.10)
    adaln = AdaLNZeroSync(D, zero_init=False)
    adaln_rms = AdaLNZeroWithRMS(D, rho=0.10, zero_init=False)

    # Sync gate values
    broadcast.gate_param.data.fill_(0.0)
    adaln.gate_param.data.fill_(0.0)
    adaln_rms.gate_param.data.fill_(0.0)

    # Fix condition
    s = torch.randn(B, D)

    # Test 1: Scale invariance
    print("\n" + "-" * 80)
    print("Test 1: Scale Invariance (delta/x ratio)")
    print("-" * 80)

    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n{'Scale':>8} | {'Broadcast':>12} | {'AdaLN':>12} | {'AdaLN+RMS':>12}")
    print("-" * 55)

    broadcast_ratios = []
    adaln_ratios = []
    adaln_rms_ratios = []

    for scale in scales:
        x = torch.randn(B, N, D) * scale

        with torch.no_grad():
            out_b = broadcast(x, s)
            out_a = adaln(x, s)
            out_ar = adaln_rms(x, s)

            x_rms = x.pow(2).mean().sqrt().item()

            delta_b = (out_b - x).pow(2).mean().sqrt().item()
            delta_a = (out_a - x).pow(2).mean().sqrt().item()
            delta_ar = (out_ar - x).pow(2).mean().sqrt().item()

            ratio_b = delta_b / x_rms
            ratio_a = delta_a / x_rms
            ratio_ar = delta_ar / x_rms

            broadcast_ratios.append(ratio_b)
            adaln_ratios.append(ratio_a)
            adaln_rms_ratios.append(ratio_ar)

        print(f"{scale:>8.2f} | {ratio_b:>12.6f} | {ratio_a:>12.6f} | {ratio_ar:>12.6f}")

    # Compute CV
    b_cv = np.std(broadcast_ratios) / np.mean(broadcast_ratios) * 100
    a_cv = np.std(adaln_ratios) / np.mean(adaln_ratios) * 100
    ar_cv = np.std(adaln_rms_ratios) / np.mean(adaln_rms_ratios) * 100

    print(f"\nCoefficient of Variation (lower = better):")
    print(f"  Broadcast:   {b_cv:>8.2f}%")
    print(f"  AdaLN:       {a_cv:>8.2f}%")
    print(f"  AdaLN+RMS:   {ar_cv:>8.2f}%")

    # Test 2: Cross-batch stability
    print("\n" + "-" * 80)
    print("Test 2: Cross-batch Stability (different conditions)")
    print("-" * 80)

    conditions = [
        ("Normal", lambda: torch.randn(B, D)),
        ("Sparse", lambda: torch.randn(B, D) * (torch.rand(B, D) > 0.8).float()),
        ("Heavy-tail", lambda: torch.randn(B, D) * 3),
        ("Uniform", lambda: torch.rand(B, D) * 2 - 1),
        ("Bimodal", lambda: torch.randn(B, D) + 2 * (torch.rand(B, 1) > 0.5).float() - 1),
    ]

    print(f"\n{'Condition':>12} | {'Broadcast':>12} | {'AdaLN':>12} | {'AdaLN+RMS':>12}")
    print("-" * 60)

    broadcast_deltas = []
    adaln_deltas = []
    adaln_rms_deltas = []

    for name, cond_fn in conditions:
        x = torch.randn(B, N, D)
        s = cond_fn()

        with torch.no_grad():
            out_b = broadcast(x, s)
            out_a = adaln(x, s)
            out_ar = adaln_rms(x, s)

            delta_b = (out_b - x).pow(2).mean().sqrt().item()
            delta_a = (out_a - x).pow(2).mean().sqrt().item()
            delta_ar = (out_ar - x).pow(2).mean().sqrt().item()

            broadcast_deltas.append(delta_b)
            adaln_deltas.append(delta_a)
            adaln_rms_deltas.append(delta_ar)

        print(f"{name:>12} | {delta_b:>12.6f} | {delta_a:>12.6f} | {delta_ar:>12.6f}")

    b_cv2 = np.std(broadcast_deltas) / np.mean(broadcast_deltas) * 100
    a_cv2 = np.std(adaln_deltas) / np.mean(adaln_deltas) * 100
    ar_cv2 = np.std(adaln_rms_deltas) / np.mean(adaln_rms_deltas) * 100

    print(f"\nCross-condition CV (lower = better):")
    print(f"  Broadcast:   {b_cv2:>8.2f}%")
    print(f"  AdaLN:       {a_cv2:>8.2f}%")
    print(f"  AdaLN+RMS:   {ar_cv2:>8.2f}%")

    # Test 3: Gradient behavior
    print("\n" + "-" * 80)
    print("Test 3: Gradient Behavior")
    print("-" * 80)

    x = torch.randn(B, N, D, requires_grad=True)
    s = torch.randn(B, D, requires_grad=True)

    methods = [
        ("Broadcast", broadcast),
        ("AdaLN", adaln),
        ("AdaLN+RMS", adaln_rms),
    ]

    print(f"\n{'Method':>12} | {'d_loss/d_x':>14} | {'d_loss/d_s':>14} | {'d_loss/d_gate':>14}")
    print("-" * 65)

    for name, model in methods:
        x_m = x.clone().detach().requires_grad_(True)
        s_m = s.clone().detach().requires_grad_(True)
        model.zero_grad()

        out = model(x_m, s_m)
        loss = out.pow(2).mean()
        loss.backward()

        print(f"{name:>12} | {x_m.grad.norm().item():>14.6f} | {s_m.grad.norm().item():>14.6f} | {model.gate_param.grad.item():>14.6f}")

    # Test 4: Energy bound analysis
    print("\n" + "-" * 80)
    print("Test 4: Absolute vs Relative Energy")
    print("-" * 80)

    scales = [0.1, 1.0, 10.0]
    s = torch.randn(B, D)

    print(f"\n{'Scale':>8} | {'Broadcast':>10} | {'AdaLN':>10} | {'AdaLN+RMS':>10} |")
    print(f"{'':>8} | {'delta':>10} | {'delta':>10} | {'delta':>10} |")
    print("-" * 55)

    for scale in scales:
        x = torch.randn(B, N, D) * scale

        with torch.no_grad():
            out_b = broadcast(x, s)
            out_a = adaln(x, s)
            out_ar = adaln_rms(x, s)

            delta_b = (out_b - x).pow(2).mean().sqrt().item()
            delta_a = (out_a - x).pow(2).mean().sqrt().item()
            delta_ar = (out_ar - x).pow(2).mean().sqrt().item()

        print(f"{scale:>8.1f} | {delta_b:>10.4f} | {delta_a:>10.4f} | {delta_ar:>10.4f} |")

    print("\nObservation:")
    print("  - Broadcast: delta scales with x (RMS matching)")
    print("  - AdaLN: delta is constant (LN normalization)")
    print("  - AdaLN+RMS: delta scales with x (RMS matching on top of LN)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
                        Broadcast    AdaLN       AdaLN+RMS
Scale Invariance CV:    {b_cv:>8.2f}%   {a_cv:>8.2f}%   {ar_cv:>8.2f}%
Cross-batch CV:         {b_cv2:>8.2f}%   {a_cv2:>8.2f}%   {ar_cv2:>8.2f}%

Analysis:
- AdaLN+RMS combines LN normalization with RMS matching
- Should have similar stability to Broadcast
- Additional LN may provide cleaner intermediate representations
    """)

    return {
        'scale_cv': {'broadcast': b_cv, 'adaln': a_cv, 'adaln_rms': ar_cv},
        'batch_cv': {'broadcast': b_cv2, 'adaln': a_cv2, 'adaln_rms': ar_cv2},
    }


def test_zero_init_behavior():
    """Test zero-init behavior for AdaLN+RMS"""
    print("\n" + "=" * 80)
    print("Test: Zero-Init Behavior for AdaLN+RMS")
    print("=" * 80)

    D = 256
    B, N = 4, 64

    adaln_rms_zero = AdaLNZeroWithRMS(D, rho=0.10, zero_init=True)
    adaln_rms_nonzero = AdaLNZeroWithRMS(D, rho=0.10, zero_init=False)

    x = torch.randn(B, N, D)
    s = torch.randn(B, D)

    with torch.no_grad():
        out_zero = adaln_rms_zero(x, s)
        out_nonzero = adaln_rms_nonzero(x, s)

        delta_zero = (out_zero - x).pow(2).mean().sqrt().item()
        delta_nonzero = (out_nonzero - x).pow(2).mean().sqrt().item()
        x_rms = x.pow(2).mean().sqrt().item()

    print(f"\nZero-init (gamma=0, beta=0):")
    print(f"  delta RMS: {delta_zero:.8f}")
    print(f"  delta/x:   {delta_zero/x_rms*100:.6f}%")

    print(f"\nNon-zero init:")
    print(f"  delta RMS: {delta_nonzero:.6f}")
    print(f"  delta/x:   {delta_nonzero/x_rms*100:.4f}%")

    # Check MLP output
    gb_zero = adaln_rms_zero.mlp(s)
    gb_nonzero = adaln_rms_nonzero.mlp(s)

    print(f"\nMLP output norms:")
    print(f"  Zero-init:     {gb_zero.norm().item():.8f}")
    print(f"  Non-zero init: {gb_nonzero.norm().item():.6f}")


def test_architectural_comparison():
    """Compare architectural properties"""
    print("\n" + "=" * 80)
    print("Architectural Comparison")
    print("=" * 80)

    D = 256

    broadcast = BroadcastBiasSync(D, rho=0.10)
    adaln = AdaLNZeroSync(D, zero_init=False)
    adaln_rms = AdaLNZeroWithRMS(D, rho=0.10, zero_init=False)

    print(f"\n{'Method':>15} | {'Parameters':>12} | {'Has LN':>8} | {'Has RMS':>8}")
    print("-" * 55)

    for name, model in [("Broadcast", broadcast), ("AdaLN", adaln), ("AdaLN+RMS", adaln_rms)]:
        params = sum(p.numel() for p in model.parameters())
        has_ln = hasattr(model, 'ln')
        has_rms = hasattr(model, 'rho')
        print(f"{name:>15} | {params:>12,} | {'Yes' if has_ln else 'No':>8} | {'Yes' if has_rms else 'No':>8}")

    print("""
Key differences:
1. Broadcast: proj(s) -> RMS match -> gate -> add
2. AdaLN: LN(x) -> MLP(s) -> FiLM -> residual
3. AdaLN+RMS: LN(x) -> MLP(s) -> FiLM -> RMS match -> add

AdaLN+RMS advantages:
- LN normalizes input before FiLM (cleaner signal)
- RMS matching controls relative energy (stable across scales)
- FiLM provides richer modulation (scale + shift)

Potential concerns:
- More computation (LN + MLP + RMS)
- Double normalization (LN + RMS) may over-constrain
    """)


def main():
    print("=" * 80)
    print("AdaLN-Zero with RMS Matching - Full Comparison")
    print("=" * 80)

    torch.manual_seed(42)

    results = test_all_three_methods()
    test_zero_init_behavior()
    test_architectural_comparison()

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    b_cv = results['scale_cv']['broadcast']
    ar_cv = results['scale_cv']['adaln_rms']
    b_cv2 = results['batch_cv']['broadcast']
    ar_cv2 = results['batch_cv']['adaln_rms']

    print(f"""
Results comparison:
                        Broadcast    AdaLN+RMS    Winner
Scale Invariance CV:    {b_cv:>8.2f}%   {ar_cv:>8.2f}%   {'Broadcast' if b_cv < ar_cv else 'AdaLN+RMS' if ar_cv < b_cv else 'Tie'}
Cross-batch CV:         {b_cv2:>8.2f}%   {ar_cv2:>8.2f}%   {'Broadcast' if b_cv2 < ar_cv2 else 'AdaLN+RMS' if ar_cv2 < b_cv2 else 'Tie'}

Conclusion:
- If AdaLN+RMS has similar CV to Broadcast: AdaLN+RMS is viable
- If AdaLN+RMS has lower CV: Consider switching
- If AdaLN+RMS has higher CV: Stay with Broadcast

Additional considerations:
1. Computational cost (AdaLN+RMS is more expensive)
2. Training dynamics (need actual training to verify)
3. Representation quality (LN may help or hurt)
    """)


if __name__ == "__main__":
    main()
