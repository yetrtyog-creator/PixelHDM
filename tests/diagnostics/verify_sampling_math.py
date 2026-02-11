"""
Verify Sampling Math - Test that the sampling formulas are correct

This creates a simple "oracle" model that knows the exact velocity,
and verifies the sampling process reconstructs the clean image.
"""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def test_sampling_with_oracle():
    """Test sampling with a perfect oracle model."""
    print("=" * 60)
    print("Testing sampling with ORACLE model")
    print("=" * 60)

    # Create synthetic data
    B, H, W, C = 1, 64, 64, 3
    device = "cpu"
    dtype = torch.float32

    # Clean image (target)
    x_clean = torch.randn(B, H, W, C, device=device, dtype=dtype)
    # Fixed noise
    noise = torch.randn(B, H, W, C, device=device, dtype=dtype)

    print(f"x_clean: mean={x_clean.mean():.4f}, std={x_clean.std():.4f}")
    print(f"noise: mean={noise.mean():.4f}, std={noise.std():.4f}")

    # Oracle velocity (what the model should learn)
    v_oracle = x_clean - noise
    print(f"v_oracle: mean={v_oracle.mean():.4f}, std={v_oracle.std():.4f}")

    # Create oracle model
    class OracleModel:
        def __init__(self, v_oracle):
            self.v_oracle = v_oracle

        def __call__(self, z, t, **kwargs):
            # Return true velocity regardless of input
            return self.v_oracle

    model = OracleModel(v_oracle)

    # Sampling parameters
    num_steps = 50
    t_eps = 0.05

    # Get timesteps (should go from t_eps to 1-t_eps)
    from src.inference.sampler.timesteps import get_timesteps
    timesteps = get_timesteps(num_steps, t_eps, device, dtype)

    print(f"\nTimesteps: {timesteps[0]:.4f} -> {timesteps[-1]:.4f}")
    print(f"dt total: {timesteps[-1] - timesteps[0]:.4f}")

    # Start from pure noise
    z = noise.clone()

    print(f"\nInitial z: mean={z.mean():.4f}, std={z.std():.4f}")

    # Euler sampling
    for i in range(num_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t

        v = model(z, t)
        z = z + dt * v

    print(f"Final z: mean={z.mean():.4f}, std={z.std():.4f}")

    # Check reconstruction error
    error = (z - x_clean).abs().mean().item()
    relative_error = error / x_clean.abs().mean().item()

    print(f"\nReconstruction error:")
    print(f"  Absolute: {error:.6f}")
    print(f"  Relative: {relative_error:.6%}")

    # Expected error should be small due to finite steps
    # With 50 steps from 0.05 to 0.95, we cover 90% of the trajectory
    # So there's ~5% error from not reaching t=1

    # Verify the formula analytically
    # z_0.05 = 0.05*x + 0.95*noise
    # z_0.95 = 0.95*x + 0.05*noise
    # With Euler: z(0.95) = z(0.05) + 0.9*v = z(0.05) + 0.9*(x-noise)
    # = 0.05*x + 0.95*noise + 0.9*x - 0.9*noise
    # = 0.95*x + 0.05*noise ✓

    expected_final = 0.95 * x_clean + 0.05 * noise
    expected_error = (z - expected_final).abs().mean().item()
    print(f"  Error vs expected z(0.95): {expected_error:.6f}")

    if relative_error < 0.15:  # Should be close to 5-10%
        print("\n[PASS] Sampling math is correct!")
        return True
    else:
        print("\n[FAIL] Sampling math has issues!")
        return False


def test_actual_sampler():
    """Test the actual sampler implementation."""
    print("\n" + "=" * 60)
    print("Testing actual sampler implementation")
    print("=" * 60)

    B, H, W, C = 1, 64, 64, 3
    device = "cpu"
    dtype = torch.float32

    x_clean = torch.randn(B, H, W, C, device=device, dtype=dtype)
    noise = torch.randn(B, H, W, C, device=device, dtype=dtype)
    v_oracle = x_clean - noise

    class OracleModel(torch.nn.Module):
        def __init__(self, v_oracle):
            super().__init__()
            self.register_buffer('v_oracle', v_oracle)

        def forward(self, z, t, **kwargs):
            return self.v_oracle

    model = OracleModel(v_oracle)

    from src.inference.sampler import create_sampler

    for method in ["euler", "heun"]:
        print(f"\nTesting {method} sampler:")

        sampler = create_sampler(method=method, num_steps=50, t_eps=0.05)

        z_final = sampler.sample(
            model=model,
            z_0=noise.clone(),
            text_embeddings=None,
            num_steps=50,
            guidance_scale=1.0,  # No CFG
        )

        error = (z_final - x_clean).abs().mean().item()
        relative_error = error / x_clean.abs().mean().item()

        print(f"  Relative error: {relative_error:.6%}")

        if relative_error < 0.15:
            print(f"  [OK] {method} PASS")
        else:
            print(f"  [FAIL] {method} FAIL")


def test_direction_check():
    """Verify sampling direction (t=0 noise -> t=1 clean)."""
    print("\n" + "=" * 60)
    print("Verifying sampling direction")
    print("=" * 60)

    # Interpolation formula: z_t = t*x + (1-t)*noise
    # At t=0: z_0 = noise
    # At t=1: z_1 = x
    # Velocity: v = x - noise (constant)
    # ODE: dz/dt = v
    # Solution: z(t) = z(0) + t*v = noise + t*(x-noise) = t*x + (1-t)*noise ✓

    print("Flow matching convention:")
    print("  z_t = t*x + (1-t)*noise")
    print("  At t=0: z = noise (start)")
    print("  At t=1: z = x (end)")
    print("")
    print("Sampling direction:")
    print("  Start: t=0.05 (near noise)")
    print("  End: t=0.95 (near clean)")
    print("  ODE: dz/dt = v = x - noise")
    print("  Euler: z_{t+dt} = z_t + dt * v")
    print("")
    print("This means:")
    print("  - We start with noise")
    print("  - Each step moves towards clean image")
    print("  - After 50 steps, we should be near x_clean")


def main():
    test_direction_check()
    success1 = test_sampling_with_oracle()
    test_actual_sampler()

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    if success1:
        print("\nSampling math is CORRECT.")
        print("If real model produces noise, the issue is likely:")
        print("  1. Model weights not loaded correctly")
        print("  2. Model not trained properly")
        print("  3. Text encoder producing bad embeddings")
        print("  4. Pooling sync gate too large")
    else:
        print("\nSampling math has ISSUES - investigate sampler code!")


if __name__ == "__main__":
    main()
