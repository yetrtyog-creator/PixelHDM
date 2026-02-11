import logging
import math

import pytest
import torch

from src.training.flow_matching.time_sampling import (
    apply_timestep_shift,
    compute_timestep_shift_mu,
    sample_logit_normal,
    get_sampling_timesteps,
)
from src.inference.sampler.timesteps import get_timesteps


def test_exponential_base_seq_is_no_shift():
    """num_tokens == base_seq_len => factor=1.0 => mu=0 (no shift)."""
    mu = compute_timestep_shift_mu(
        num_tokens=256, shift_type="exponential", fixed_shift=3.0,
        base_seq_len=256, max_seq_len=4096,
    )
    assert mu == pytest.approx(0.0, abs=1e-6)


def test_exponential_max_seq_is_full_shift():
    """num_tokens == max_seq_len => factor=fixed_shift => mu=log(3.0)."""
    mu = compute_timestep_shift_mu(
        num_tokens=4096, shift_type="exponential", fixed_shift=3.0,
        base_seq_len=256, max_seq_len=4096,
    )
    assert mu == pytest.approx(math.log(3.0), rel=1e-6, abs=1e-6)


def test_exponential_mu_monotonic_with_tokens():
    """mu should increase with num_tokens between base and max."""
    base, mid, max_ = 256, 2048, 4096
    mu_base = compute_timestep_shift_mu(
        num_tokens=base, shift_type="exponential", fixed_shift=3.0,
        base_seq_len=256, max_seq_len=4096,
    )
    mu_mid = compute_timestep_shift_mu(
        num_tokens=mid, shift_type="exponential", fixed_shift=3.0,
        base_seq_len=256, max_seq_len=4096,
    )
    mu_max = compute_timestep_shift_mu(
        num_tokens=max_, shift_type="exponential", fixed_shift=3.0,
        base_seq_len=256, max_seq_len=4096,
    )
    assert mu_base < mu_mid < mu_max


def test_compute_timestep_shift_mu_linear_clamp():
    mu = compute_timestep_shift_mu(
        num_tokens=1_000_000,
        shift_type="linear",
        base_shift=0.5,
        max_shift=1.15,
        base_seq_len=256,
        max_seq_len=4096,
        clamp_linear=True,
    )
    assert mu == pytest.approx(1.15, rel=1e-6, abs=1e-6)


def test_apply_timestep_shift_direction():
    """At max_seq_len, mu=log(3.0), shift should move t=0.9 toward noise."""
    t_p = torch.tensor([0.9], dtype=torch.float32)
    sigma = 1.0 - t_p
    mu = compute_timestep_shift_mu(
        num_tokens=4096, shift_type="exponential", fixed_shift=3.0,
        base_seq_len=256, max_seq_len=4096,
    )
    sigma_shifted = apply_timestep_shift(sigma, mu, "exponential")
    t_p_shifted = 1.0 - sigma_shifted
    assert t_p_shifted.item() == pytest.approx(0.75, rel=1e-3, abs=1e-3)


def test_shift_config_ignores_extra_keys_warns(caplog):
    caplog.set_level(logging.WARNING)
    shift_config = {
        "shift_type": "exponential",
        "fixed_shift": 1.0,
        "extra_key": 123,
    }
    torch.manual_seed(0)
    _ = sample_logit_normal(
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        p_mean=0.0,
        p_std=1.0,
        t_eps=0.05,
        num_tokens=256,
        shift_config=shift_config,
    )
    assert any("Ignoring unknown shift_config keys" in rec.message for rec in caplog.records)


def test_shift_changes_timestep_mean():
    """At max_seq_len (4096), DTS shift should lower the mean timestep."""
    shift_config = {
        "shift_type": "exponential",
        "fixed_shift": 3.0,
        "base_seq_len": 256,
        "max_seq_len": 4096,
    }
    t_base = get_sampling_timesteps(
        num_steps=50,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_eps=0.05,
        use_logit_normal=True,
        p_mean=0.0,
        p_std=1.0,
    )
    t_shifted = get_sampling_timesteps(
        num_steps=50,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_eps=0.05,
        use_logit_normal=True,
        p_mean=0.0,
        p_std=1.0,
        num_tokens=4096,
        shift_config=shift_config,
    )
    assert t_shifted.mean().item() < t_base.mean().item()


def test_inference_get_timesteps_shift_applied():
    """At max_seq_len (4096), inference DTS shift should lower the mean timestep."""
    shift_config = {
        "shift_type": "exponential",
        "fixed_shift": 3.0,
        "base_seq_len": 256,
        "max_seq_len": 4096,
    }
    t_base = get_timesteps(
        num_steps=10,
        t_eps=0.05,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_logit_normal=True,
        p_mean=0.0,
        p_std=1.0,
    )
    t_shifted = get_timesteps(
        num_steps=10,
        t_eps=0.05,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_logit_normal=True,
        p_mean=0.0,
        p_std=1.0,
        num_tokens=4096,
        shift_config=shift_config,
    )
    assert t_shifted.mean().item() < t_base.mean().item()


def test_exponential_raises_when_max_eq_base():
    """max_seq_len == base_seq_len should raise ValueError."""
    with pytest.raises(ValueError, match="max_seq_len must differ"):
        compute_timestep_shift_mu(
            num_tokens=256, shift_type="exponential", fixed_shift=3.0,
            base_seq_len=256, max_seq_len=256,
        )


def test_exponential_rejects_non_positive_fixed_shift():
    """fixed_shift <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="fixed_shift must be > 0"):
        compute_timestep_shift_mu(
            num_tokens=256, shift_type="exponential", fixed_shift=0.0,
            base_seq_len=256, max_seq_len=4096,
        )
    with pytest.raises(ValueError, match="fixed_shift must be > 0"):
        compute_timestep_shift_mu(
            num_tokens=256, shift_type="exponential", fixed_shift=-1.0,
            base_seq_len=256, max_seq_len=4096,
        )
