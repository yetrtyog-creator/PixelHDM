"""
Test: Bottleneck Dimension Configuration Pass-through

Verifies that bottleneck_dim can be:
1. Auto-calculated from patch_size (default)
2. Set explicitly in PixelHDMConfig
3. Set from train_config.yaml and properly passed through

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-24
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_bottleneck_dim_default_calculation():
    """Test that bottleneck_dim is auto-calculated when not provided."""
    from src.config import PixelHDMConfig

    # Test 1: patch_size=16 -> bottleneck_dim = 16^2 // 4 = 64
    config = PixelHDMConfig(patch_size=16)
    assert config.bottleneck_dim == 64, \
        f"Expected bottleneck_dim=64 for patch_size=16, got {config.bottleneck_dim}"
    print(f"[PASS] patch_size=16 -> bottleneck_dim={config.bottleneck_dim} (expected 64)")

    # Test 2: patch_size=8 -> bottleneck_dim = 8^2 // 4 = 16
    config = PixelHDMConfig(patch_size=8)
    assert config.bottleneck_dim == 16, \
        f"Expected bottleneck_dim=16 for patch_size=8, got {config.bottleneck_dim}"
    print(f"[PASS] patch_size=8 -> bottleneck_dim={config.bottleneck_dim} (expected 16)")

    # Test 3: patch_size=32 -> bottleneck_dim = 32^2 // 4 = 256
    config = PixelHDMConfig(patch_size=32)
    assert config.bottleneck_dim == 256, \
        f"Expected bottleneck_dim=256 for patch_size=32, got {config.bottleneck_dim}"
    print(f"[PASS] patch_size=32 -> bottleneck_dim={config.bottleneck_dim} (expected 256)")


def test_bottleneck_dim_explicit_override():
    """Test that explicit bottleneck_dim overrides auto-calculation."""
    from src.config import PixelHDMConfig

    # Test explicit override
    config = PixelHDMConfig(patch_size=16, bottleneck_dim=128)
    assert config.bottleneck_dim == 128, \
        f"Expected bottleneck_dim=128 (explicit), got {config.bottleneck_dim}"
    print(f"[PASS] Explicit bottleneck_dim=128 overrides default")

    # Test with small value
    config = PixelHDMConfig(patch_size=16, bottleneck_dim=16)
    assert config.bottleneck_dim == 16, \
        f"Expected bottleneck_dim=16 (explicit), got {config.bottleneck_dim}"
    print(f"[PASS] Explicit bottleneck_dim=16 overrides default")

    # Test with large value
    config = PixelHDMConfig(patch_size=16, bottleneck_dim=512)
    assert config.bottleneck_dim == 512, \
        f"Expected bottleneck_dim=512 (explicit), got {config.bottleneck_dim}"
    print(f"[PASS] Explicit bottleneck_dim=512 overrides default")


def test_bottleneck_dim_from_yaml():
    """Test that bottleneck_dim is correctly parsed from YAML config."""
    from src.config.parsers import parse_model_config

    # Minimal valid config to pass validation
    # mRoPE: text_dim + img_h_dim + img_w_dim = head_dim
    # Default head_dim=64, so 16 + 24 + 24 = 64
    base_config = {
        "hidden_dim": 512,
        "num_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 64,
        "mrope_text_dim": 16,
        "mrope_img_h_dim": 24,
        "mrope_img_w_dim": 24,
    }

    # Test 1: YAML with explicit bottleneck_dim
    yaml_data = {
        "model": {
            **base_config,
            "patch_size": 16,
            "bottleneck_dim": 64,
        }
    }
    config = parse_model_config(yaml_data)
    assert config.bottleneck_dim == 64, \
        f"Expected bottleneck_dim=64 from YAML, got {config.bottleneck_dim}"
    print(f"[PASS] YAML bottleneck_dim=64 correctly parsed")

    # Test 2: YAML without bottleneck_dim (should use default)
    yaml_data = {
        "model": {
            **base_config,
            "patch_size": 16,
        }
    }
    config = parse_model_config(yaml_data)
    assert config.bottleneck_dim == 64, \
        f"Expected bottleneck_dim=64 (auto), got {config.bottleneck_dim}"
    print(f"[PASS] YAML without bottleneck_dim -> auto-calculated={config.bottleneck_dim}")

    # Test 3: YAML with different patch_size
    yaml_data = {
        "model": {
            **base_config,
            "patch_size": 8,
        }
    }
    config = parse_model_config(yaml_data)
    assert config.bottleneck_dim == 16, \
        f"Expected bottleneck_dim=16 (auto for patch_size=8), got {config.bottleneck_dim}"
    print(f"[PASS] YAML patch_size=8 -> auto-calculated bottleneck_dim={config.bottleneck_dim}")


def test_bottleneck_dim_in_model():
    """Test that bottleneck_dim is correctly used in model creation."""
    from src.config import PixelHDMConfig
    from src.models.pixelhdm import PixelHDM

    # Create config with explicit bottleneck_dim
    # Need to set mRoPE dims to match head_dim for validation
    config = PixelHDMConfig(
        hidden_dim=512,
        patch_size=16,
        patch_layers=2,
        pixel_layers=1,
        num_heads=8,
        num_kv_heads=4,
        head_dim=64,
        mrope_text_dim=16,
        mrope_img_h_dim=24,
        mrope_img_w_dim=24,
        bottleneck_dim=64,
        text_hidden_size=512,
        repa_enabled=False,
        freq_loss_enabled=False,
        use_flash_attention=False,
        use_gradient_checkpointing=False,
    )

    # Create model
    model = PixelHDM(config=config)

    # Verify patch_embed has correct bottleneck_dim
    patch_embed = model.patch_embed
    assert hasattr(patch_embed, 'bottleneck_dim'), \
        "PatchEmbedding should have bottleneck_dim attribute"
    assert patch_embed.bottleneck_dim == 64, \
        f"Expected patch_embed.bottleneck_dim=64, got {patch_embed.bottleneck_dim}"
    print(f"[PASS] Model patch_embed.bottleneck_dim={patch_embed.bottleneck_dim}")

    # Verify proj_down shape
    proj_down = patch_embed.proj_down
    input_dim = 16 * 16 * 3  # patch_size^2 * channels = 768
    assert proj_down.weight.shape == (64, input_dim), \
        f"Expected proj_down shape (64, {input_dim}), got {proj_down.weight.shape}"
    print(f"[PASS] proj_down.weight.shape={proj_down.weight.shape} (expected (64, 768))")


def test_real_train_config():
    """Test parsing the actual train_config.yaml file."""
    import yaml

    config_path = Path(__file__).parent.parent.parent / "configs" / "train_config.yaml"

    if not config_path.exists():
        print(f"[SKIP] train_config.yaml not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    from src.config.parsers import parse_model_config
    config = parse_model_config(yaml_data)

    print(f"\n--- Real train_config.yaml parsing ---")
    print(f"patch_size: {config.patch_size}")
    print(f"bottleneck_dim: {config.bottleneck_dim}")
    print(f"hidden_dim: {config.hidden_dim}")

    # Verify bottleneck_dim is set (either explicitly or auto-calculated)
    assert config.bottleneck_dim is not None, "bottleneck_dim should not be None"
    assert config.bottleneck_dim > 0, "bottleneck_dim should be positive"

    # Check if it matches expected value from YAML
    model_section = yaml_data.get("model", {})
    if "bottleneck_dim" in model_section:
        expected = model_section["bottleneck_dim"]
        assert config.bottleneck_dim == expected, \
            f"Expected bottleneck_dim={expected} from YAML, got {config.bottleneck_dim}"
        print(f"[PASS] Real train_config.yaml: bottleneck_dim={config.bottleneck_dim} (from YAML)")
    else:
        expected = config.patch_size ** 2 // 4
        assert config.bottleneck_dim == expected, \
            f"Expected auto-calculated bottleneck_dim={expected}, got {config.bottleneck_dim}"
        print(f"[PASS] Real train_config.yaml: bottleneck_dim={config.bottleneck_dim} (auto-calculated)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Bottleneck Dimension Configuration Tests")
    print("=" * 60)

    print("\n--- Test 1: Default Calculation ---")
    test_bottleneck_dim_default_calculation()

    print("\n--- Test 2: Explicit Override ---")
    test_bottleneck_dim_explicit_override()

    print("\n--- Test 3: YAML Parsing ---")
    test_bottleneck_dim_from_yaml()

    print("\n--- Test 4: Model Integration ---")
    test_bottleneck_dim_in_model()

    print("\n--- Test 5: Real train_config.yaml ---")
    test_real_train_config()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
