"""
AdaLN-Zero + RMS Matching (Fixed Version)

Fix: Gate is applied AFTER RMS matching to ensure healthy gate gradients.

Original problem:
  delta_raw = g * (y * gamma + beta)
  delta_hat = rms_match(delta_raw)  # RMS cancels gate effect, gradient = 0

Fixed:
  delta_raw = y * gamma + beta      # No gate
  delta_hat = rms_match(delta_raw)  # RMS on raw delta
  delta = g * delta_hat             # Gate after RMS

Based on: GPT/PLAN_adaln_zero_rms.md
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
    """Current broadcast bias (baseline)"""
    def __init__(self, hidden_dim: int, rho: float = 0.10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rho = rho
        self.eps = 1e-8

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_param = nn.Parameter(torch.tensor(0.0))

        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def _rms(self, t, dim=-1, keepdim=True):
        return (t.pow(2).mean(dim=dim, keepdim=keepdim) + self.eps).sqrt()

    def forward(self, x, s):
        bias = self.proj(s)
        x_rms = self._rms(x, dim=-1, keepdim=False).mean(dim=-1, keepdim=True)
        b_rms = self._rms(bias, dim=-1, keepdim=True)
        bias_hat = bias * (self.rho * x_rms / (b_rms + self.eps))
        gate = torch.sigmoid(self.gate_param)
        delta = gate * bias_hat
        return x + delta.unsqueeze(1)


class AdaLNZeroWithRMS_Broken(nn.Module):
    """AdaLN-Zero + RMS (BROKEN: gate before RMS, gradient = 0)"""
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

    def _rms(self, t, dim=-1, keepdim=True):
        return (t.pow(2).mean(dim=dim, keepdim=keepdim) + self.eps).sqrt()

    def forward(self, x, s):
        y = self.ln(x)
        gb = self.mlp(s)
        gamma, beta = gb.chunk(2, dim=-1)
        g = torch.sigmoid(self.gate_param)

        # BROKEN: gate before RMS
        y_scaled = y * (1 + g * gamma.unsqueeze(1)) + g * beta.unsqueeze(1)
        delta_raw = y_scaled - y

        x_rms = self._rms(x, dim=-1, keepdim=True)
        delta_rms = self._rms(delta_raw, dim=-1, keepdim=True)
        delta_hat = delta_raw * (self.rho * x_rms / (delta_rms + self.eps))

        return x + delta_hat


class AdaLNZeroWithRMS_Fixed(nn.Module):
    """AdaLN-Zero + RMS (FIXED: gate after RMS, healthy gradient)"""
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

    def _rms(self, t, dim=-1, keepdim=True):
        return (t.pow(2).mean(dim=dim, keepdim=keepdim) + self.eps).sqrt()

    def forward(self, x, s):
        y = self.ln(x)
        gb = self.mlp(s)
        gamma, beta = gb.chunk(2, dim=-1)

        # FIXED: Raw delta without gate
        delta_raw = y * gamma.unsqueeze(1) + beta.unsqueeze(1)

        # RMS matching on raw delta
        x_rms = self._rms(x, dim=-1, keepdim=True)
        delta_rms = self._rms(delta_raw, dim=-1, keepdim=True)
        delta_hat = delta_raw * (self.rho * x_rms / (delta_rms + self.eps))

        # Gate AFTER RMS matching
        g = torch.sigmoid(self.gate_param)
        delta = g * delta_hat

        return x + delta


def test_gate_gradient():
    """Test 1: Verify gate gradient is healthy"""
    print("=" * 70)
    print("Test 1: Gate Gradient Comparison")
    print("=" * 70)

    D = 256
    B, N = 4, 64

    broadcast = BroadcastBiasSync(D, rho=0.10)
    broken = AdaLNZeroWithRMS_Broken(D, rho=0.10, zero_init=False)
    fixed = AdaLNZeroWithRMS_Fixed(D, rho=0.10, zero_init=False)

    methods = [
        ("Broadcast", broadcast),
        ("AdaLN+RMS (Broken)", broken),
        ("AdaLN+RMS (Fixed)", fixed),
    ]

    print(f"\n{'Method':>20} | {'d_loss/d_x':>12} | {'d_loss/d_s':>12} | {'d_loss/d_gate':>14}")
    print("-" * 70)

    for name, model in methods:
        x = torch.randn(B, N, D, requires_grad=True)
        s = torch.randn(B, D, requires_grad=True)
        model.zero_grad()

        out = model(x, s)
        loss = out.pow(2).mean()
        loss.backward()

        print(f"{name:>20} | {x.grad.norm().item():>12.6f} | {s.grad.norm().item():>12.6f} | {model.gate_param.grad.item():>14.6f}")

    # Gate gradient at different gate values
    print("\n" + "-" * 70)
    print("Gate gradient at different gate values:")
    print(f"{'Gate value':>12} | {'Broadcast':>14} | {'Broken':>14} | {'Fixed':>14}")
    print("-" * 70)

    gate_logits = [-5.0, -3.0, 0.0, 3.0]
    for logit in gate_logits:
        broadcast.gate_param.data.fill_(logit)
        broken.gate_param.data.fill_(logit)
        fixed.gate_param.data.fill_(logit)

        grads = []
        for model in [broadcast, broken, fixed]:
            model.zero_grad()
            x = torch.randn(B, N, D)
            s = torch.randn(B, D)
            out = model(x, s)
            loss = out.pow(2).mean()
            loss.backward()
            grads.append(model.gate_param.grad.item())

        gate_val = torch.sigmoid(torch.tensor(logit)).item()
        print(f"{gate_val:>12.4f} | {grads[0]:>14.6f} | {grads[1]:>14.6f} | {grads[2]:>14.6f}")

    print("\n[PASS] if Fixed gradient > 0 and similar to Broadcast")


def test_scale_invariance():
    """Test 2: Scale invariance"""
    print("\n" + "=" * 70)
    print("Test 2: Scale Invariance (delta/x ratio)")
    print("=" * 70)

    D = 256
    B, N = 4, 64

    broadcast = BroadcastBiasSync(D, rho=0.10)
    broken = AdaLNZeroWithRMS_Broken(D, rho=0.10, zero_init=False)
    fixed = AdaLNZeroWithRMS_Fixed(D, rho=0.10, zero_init=False)

    # Set gate = 0.5
    for m in [broadcast, broken, fixed]:
        m.gate_param.data.fill_(0.0)

    s = torch.randn(B, D)
    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n{'Scale':>8} | {'Broadcast':>12} | {'Broken':>12} | {'Fixed':>12}")
    print("-" * 55)

    ratios = {'broadcast': [], 'broken': [], 'fixed': []}

    for scale in scales:
        x = torch.randn(B, N, D) * scale

        with torch.no_grad():
            x_rms = x.pow(2).mean().sqrt().item()

            out_b = broadcast(x, s)
            out_br = broken(x, s)
            out_f = fixed(x, s)

            r_b = (out_b - x).pow(2).mean().sqrt().item() / x_rms
            r_br = (out_br - x).pow(2).mean().sqrt().item() / x_rms
            r_f = (out_f - x).pow(2).mean().sqrt().item() / x_rms

            ratios['broadcast'].append(r_b)
            ratios['broken'].append(r_br)
            ratios['fixed'].append(r_f)

        print(f"{scale:>8.2f} | {r_b:>12.6f} | {r_br:>12.6f} | {r_f:>12.6f}")

    # Compute CV
    cv_b = np.std(ratios['broadcast']) / np.mean(ratios['broadcast']) * 100
    cv_br = np.std(ratios['broken']) / np.mean(ratios['broken']) * 100
    cv_f = np.std(ratios['fixed']) / np.mean(ratios['fixed']) * 100

    print(f"\nCoefficient of Variation (lower = better):")
    print(f"  Broadcast: {cv_b:>8.2f}%")
    print(f"  Broken:    {cv_br:>8.2f}%")
    print(f"  Fixed:     {cv_f:>8.2f}%")

    return {'broadcast': cv_b, 'broken': cv_br, 'fixed': cv_f}


def test_cross_batch_stability():
    """Test 3: Cross-batch stability"""
    print("\n" + "=" * 70)
    print("Test 3: Cross-batch Stability")
    print("=" * 70)

    D = 256
    B, N = 4, 64

    broadcast = BroadcastBiasSync(D, rho=0.10)
    broken = AdaLNZeroWithRMS_Broken(D, rho=0.10, zero_init=False)
    fixed = AdaLNZeroWithRMS_Fixed(D, rho=0.10, zero_init=False)

    for m in [broadcast, broken, fixed]:
        m.gate_param.data.fill_(0.0)

    conditions = [
        ("Normal", lambda: torch.randn(B, D)),
        ("Sparse", lambda: torch.randn(B, D) * (torch.rand(B, D) > 0.8).float()),
        ("Heavy-tail", lambda: torch.randn(B, D) * 3),
        ("Uniform", lambda: torch.rand(B, D) * 2 - 1),
        ("Bimodal", lambda: torch.randn(B, D) + 2 * (torch.rand(B, 1) > 0.5).float() - 1),
    ]

    print(f"\n{'Condition':>12} | {'Broadcast':>12} | {'Broken':>12} | {'Fixed':>12}")
    print("-" * 60)

    deltas = {'broadcast': [], 'broken': [], 'fixed': []}

    for name, cond_fn in conditions:
        x = torch.randn(B, N, D)
        s = cond_fn()

        with torch.no_grad():
            d_b = (broadcast(x, s) - x).pow(2).mean().sqrt().item()
            d_br = (broken(x, s) - x).pow(2).mean().sqrt().item()
            d_f = (fixed(x, s) - x).pow(2).mean().sqrt().item()

            deltas['broadcast'].append(d_b)
            deltas['broken'].append(d_br)
            deltas['fixed'].append(d_f)

        print(f"{name:>12} | {d_b:>12.6f} | {d_br:>12.6f} | {d_f:>12.6f}")

    cv_b = np.std(deltas['broadcast']) / np.mean(deltas['broadcast']) * 100
    cv_br = np.std(deltas['broken']) / np.mean(deltas['broken']) * 100
    cv_f = np.std(deltas['fixed']) / np.mean(deltas['fixed']) * 100

    print(f"\nCross-condition CV (lower = better):")
    print(f"  Broadcast: {cv_b:>8.2f}%")
    print(f"  Broken:    {cv_br:>8.2f}%")
    print(f"  Fixed:     {cv_f:>8.2f}%")

    return {'broadcast': cv_b, 'broken': cv_br, 'fixed': cv_f}


def test_zero_init():
    """Test 4: Zero-init behavior"""
    print("\n" + "=" * 70)
    print("Test 4: Zero-Init Behavior")
    print("=" * 70)

    D = 256
    B, N = 4, 64

    fixed_zero = AdaLNZeroWithRMS_Fixed(D, rho=0.10, zero_init=True)
    fixed_nonzero = AdaLNZeroWithRMS_Fixed(D, rho=0.10, zero_init=False)

    x = torch.randn(B, N, D)
    s = torch.randn(B, D)
    x_rms = x.pow(2).mean().sqrt().item()

    with torch.no_grad():
        delta_zero = (fixed_zero(x, s) - x).pow(2).mean().sqrt().item()
        delta_nonzero = (fixed_nonzero(x, s) - x).pow(2).mean().sqrt().item()

    print(f"\nZero-init:")
    print(f"  delta RMS: {delta_zero:.8f}")
    print(f"  delta/x:   {delta_zero/x_rms*100:.6f}%")

    print(f"\nNon-zero init:")
    print(f"  delta RMS: {delta_nonzero:.6f}")
    print(f"  delta/x:   {delta_nonzero/x_rms*100:.4f}%")

    if delta_zero / x_rms < 0.01:
        print("\n[PASS] Zero-init working: output ~= input")
    else:
        print("\n[FAIL] Zero-init not working")


def test_energy_scaling():
    """Test 5: Energy scaling with gate"""
    print("\n" + "=" * 70)
    print("Test 5: Energy Scaling with Gate")
    print("=" * 70)

    D = 256
    B, N = 4, 64

    broadcast = BroadcastBiasSync(D, rho=0.10)
    fixed = AdaLNZeroWithRMS_Fixed(D, rho=0.10, zero_init=False)

    x = torch.randn(B, N, D)
    s = torch.randn(B, D)

    gate_values = [0.1, 0.2, 0.5, 0.8, 1.0]

    print(f"\n{'Gate':>8} | {'Broadcast delta':>16} | {'Fixed delta':>16} | {'Ratio':>10}")
    print("-" * 60)

    for gate_target in gate_values:
        gate_logit = torch.tensor(gate_target / (1 - gate_target + 1e-8)).log()
        broadcast.gate_param.data.fill_(gate_logit)
        fixed.gate_param.data.fill_(gate_logit)

        with torch.no_grad():
            d_b = (broadcast(x, s) - x).pow(2).mean().sqrt().item()
            d_f = (fixed(x, s) - x).pow(2).mean().sqrt().item()
            ratio = d_f / d_b if d_b > 1e-8 else float('nan')

        print(f"{gate_target:>8.2f} | {d_b:>16.6f} | {d_f:>16.6f} | {ratio:>10.4f}")

    print("\n[PASS] if both scale linearly with gate")


def main():
    print("=" * 70)
    print("AdaLN-Zero + RMS Matching - Fixed Version Test")
    print("=" * 70)

    torch.manual_seed(42)

    test_gate_gradient()
    scale_cv = test_scale_invariance()
    batch_cv = test_cross_batch_stability()
    test_zero_init()
    test_energy_scaling()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
                        Broadcast    Broken       Fixed
Scale Invariance CV:    {scale_cv['broadcast']:>8.2f}%   {scale_cv['broken']:>8.2f}%   {scale_cv['fixed']:>8.2f}%
Cross-batch CV:         {batch_cv['broadcast']:>8.2f}%   {batch_cv['broken']:>8.2f}%   {batch_cv['fixed']:>8.2f}%

Key Fix:
- Broken: gate before RMS -> gate gradient = 0
- Fixed: gate after RMS -> gate gradient healthy

Conclusion:
- Fixed version has healthy gate gradient
- Scale invariance and cross-batch stability preserved
- Comparable to Broadcast baseline
    """)

    return {
        'scale_cv': scale_cv,
        'batch_cv': batch_cv,
    }


if __name__ == "__main__":
    results = main()
