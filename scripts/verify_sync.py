#!/usr/bin/env python3
"""
PixelHDM-RPEA-DinoV3 - Environment Configuration Sync Verification

This script verifies that all configuration files and scripts are in sync
WITHOUT executing any installations.

Usage:
    python scripts/verify_sync.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Project root (relative to this script)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def print_result(name: str, passed: bool, details: str = "") -> None:
    """Print a test result."""
    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} {name}")
    if details and not passed:
        print(f"       -> {details}")


def check_file_exists(path: Path) -> Tuple[bool, str]:
    """Check if a file exists."""
    if path.exists():
        return True, ""
    return False, f"File not found: {path}"


def check_venv_name_in_file(filepath: Path, expected_name: str) -> Tuple[bool, str]:
    """Check if a file references the correct venv name."""
    if not filepath.exists():
        return False, f"File not found: {filepath}"

    content = filepath.read_text(encoding='utf-8', errors='ignore')

    # Look for VENV_PATH pattern
    pattern = r'VENV_PATH.*\.venv\\([a-zA-Z0-9_-]+)'
    matches = re.findall(pattern, content)

    if not matches:
        return False, f"No VENV_PATH found in {filepath.name}"

    for match in matches:
        if match != expected_name:
            return False, f"Found '{match}' instead of '{expected_name}' in {filepath.name}"

    return True, ""


def check_no_absolute_paths(filepath: Path, exceptions: List[str] = None) -> Tuple[bool, str]:
    """Check that a file doesn't contain absolute Windows paths (except allowed ones)."""
    if exceptions is None:
        exceptions = []

    if not filepath.exists():
        return False, f"File not found: {filepath}"

    content = filepath.read_text(encoding='utf-8', errors='ignore')

    # Pattern for Windows absolute paths like C:\, D:\, G:\
    pattern = r'[A-Z]:\\[^\s\'"]*'
    matches = re.findall(pattern, content)

    # Filter out exceptions
    violations = []
    for match in matches:
        is_exception = any(exc in filepath.name for exc in exceptions)
        if not is_exception:
            violations.append(match)

    if violations:
        return False, f"Absolute paths found: {violations[:3]}..."  # Show first 3

    return True, ""


def check_yaml_syntax(filepath: Path) -> Tuple[bool, str]:
    """Basic YAML syntax check (without importing yaml)."""
    if not filepath.exists():
        return False, f"File not found: {filepath}"

    content = filepath.read_text(encoding='utf-8', errors='ignore')

    # Basic checks
    if not content.strip():
        return False, "File is empty"

    # Check for common YAML issues
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        # Check for tabs (YAML should use spaces)
        if '\t' in line and not line.strip().startswith('#'):
            return False, f"Line {i}: Tab character found (use spaces in YAML)"

    return True, ""


def verify_batch_files() -> Dict[str, Tuple[bool, str]]:
    """Verify all batch files reference correct venv name."""
    expected_name = "pixelHDM-env"

    batch_files = [
        PROJECT_ROOT / "scripts" / "setup_env.bat",
        PROJECT_ROOT / "scripts" / "activate.bat",
        PROJECT_ROOT / "scripts" / "train.bat",
        PROJECT_ROOT / "scripts" / "inference.bat",
        PROJECT_ROOT / "scripts" / "verify_env.bat",
        PROJECT_ROOT / "scripts" / "run_tests.bat",
    ]

    results = {}
    for bf in batch_files:
        results[bf.name] = check_venv_name_in_file(bf, expected_name)

    return results


def verify_config_files() -> Dict[str, Tuple[bool, str]]:
    """Verify configuration files exist and have valid syntax."""
    results = {}

    # Required config files
    configs = [
        PROJECT_ROOT / "configs" / "env_config.yaml",
        PROJECT_ROOT / "configs" / "train_config.yaml",
        PROJECT_ROOT / "configs" / "data_config.yaml",
        PROJECT_ROOT / "requirements.txt",
    ]

    for cfg in configs:
        exists, msg = check_file_exists(cfg)
        results[f"{cfg.name} exists"] = (exists, msg)

        if exists and cfg.suffix == ".yaml":
            syntax_ok, syntax_msg = check_yaml_syntax(cfg)
            results[f"{cfg.name} syntax"] = (syntax_ok, syntax_msg)

    return results


def verify_relative_paths() -> Dict[str, Tuple[bool, str]]:
    """Verify no absolute paths in config files (except data_config.yaml)."""
    results = {}

    # Files that should NOT have absolute paths
    check_files = [
        PROJECT_ROOT / "configs" / "env_config.yaml",
        PROJECT_ROOT / "configs" / "train_config.yaml",
    ]

    for cf in check_files:
        results[f"{cf.name} relative paths"] = check_no_absolute_paths(cf)

    # data_config.yaml is ALLOWED to have absolute paths
    data_cfg = PROJECT_ROOT / "configs" / "data_config.yaml"
    if data_cfg.exists():
        results["data_config.yaml (abs allowed)"] = (True, "Absolute paths allowed")

    return results


def verify_docs_sync() -> Dict[str, Tuple[bool, str]]:
    """Verify documentation is in sync with actual config."""
    results = {}

    readme = PROJECT_ROOT / "publicdocs" / "README.md"
    if readme.exists():
        content = readme.read_text(encoding='utf-8', errors='ignore')

        # Check for correct venv name
        if "pixelHDM-env" in content:
            results["README.md venv name"] = (True, "")
        else:
            results["README.md venv name"] = (False, "pixelHDM-env not found in README")

        # Check for DINOv3 note
        if "DINOv3" in content and ("授權" in content or "license" in content.lower()):
            results["README.md DINOv3 note"] = (True, "")
        else:
            results["README.md DINOv3 note"] = (False, "DINOv3 license note not found")
    else:
        results["README.md exists"] = (False, f"Not found: {readme}")

    return results


def verify_dinov3_path() -> Dict[str, Tuple[bool, str]]:
    """Check DINOv3 weights path is correctly referenced."""
    results = {}

    dinov3_path = PROJECT_ROOT / "Dinov3" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    if dinov3_path.exists():
        results["DINOv3 weights"] = (True, "Found")
    else:
        results["DINOv3 weights"] = (False, "NOT FOUND (requires manual download with Meta license)")

    # Check if referenced in env_config.yaml
    env_cfg = PROJECT_ROOT / "configs" / "env_config.yaml"
    if env_cfg.exists():
        content = env_cfg.read_text(encoding='utf-8', errors='ignore')
        if "dinov3_vitb16_pretrain_lvd1689m" in content:
            results["DINOv3 in env_config"] = (True, "")
        else:
            results["DINOv3 in env_config"] = (False, "DINOv3 path not found in env_config.yaml")

    return results


def verify_venv_directory() -> Dict[str, Tuple[bool, str]]:
    """Check the actual venv directory status."""
    results = {}

    expected_path = PROJECT_ROOT / ".venv" / "pixelHDM-env"
    old_path = PROJECT_ROOT / ".venv" / "jit-lumina2-env"

    if expected_path.exists():
        results["pixelHDM-env directory"] = (True, "Found")
    else:
        if old_path.exists():
            results["pixelHDM-env directory"] = (
                False,
                f"NOT FOUND - Old directory exists at {old_path.name}. "
                "Please rename: jit-lumina2-env -> pixelHDM-env"
            )
        else:
            results["pixelHDM-env directory"] = (
                False,
                "NOT FOUND - Run scripts\\setup_env.bat to create"
            )

    return results


def main():
    """Run all verification checks."""
    print_header("PixelHDM-RPEA-DinoV3 Environment Sync Verification")
    print(f"Project root: {PROJECT_ROOT}")

    all_passed = True
    total_checks = 0
    passed_checks = 0

    # 1. Batch files
    print_header("1. Batch File Verification")
    batch_results = verify_batch_files()
    for name, (passed, msg) in batch_results.items():
        print_result(name, passed, msg)
        total_checks += 1
        if passed:
            passed_checks += 1
        else:
            all_passed = False

    # 2. Config files
    print_header("2. Configuration Files")
    config_results = verify_config_files()
    for name, (passed, msg) in config_results.items():
        print_result(name, passed, msg)
        total_checks += 1
        if passed:
            passed_checks += 1
        else:
            all_passed = False

    # 3. Relative paths
    print_header("3. Relative Path Verification")
    path_results = verify_relative_paths()
    for name, (passed, msg) in path_results.items():
        print_result(name, passed, msg)
        total_checks += 1
        if passed:
            passed_checks += 1
        else:
            all_passed = False

    # 4. Documentation
    print_header("4. Documentation Sync")
    doc_results = verify_docs_sync()
    for name, (passed, msg) in doc_results.items():
        print_result(name, passed, msg)
        total_checks += 1
        if passed:
            passed_checks += 1
        else:
            all_passed = False

    # 5. DINOv3
    print_header("5. DINOv3 Verification")
    dinov3_results = verify_dinov3_path()
    for name, (passed, msg) in dinov3_results.items():
        print_result(name, passed, msg)
        total_checks += 1
        if passed:
            passed_checks += 1
        # DINOv3 weights missing is a warning, not failure

    # 6. VEnv Directory
    print_header("6. Virtual Environment Directory")
    venv_results = verify_venv_directory()
    for name, (passed, msg) in venv_results.items():
        print_result(name, passed, msg)
        total_checks += 1
        if passed:
            passed_checks += 1
        else:
            all_passed = False

    # Summary
    print_header("Summary")
    print(f"  Checks passed: {passed_checks}/{total_checks}")

    if all_passed:
        print("\n  [SUCCESS] All configuration files are in sync!")
        return 0
    else:
        print("\n  [ACTION REQUIRED] Some checks failed. See details above.")

        # Check if it's just the venv rename
        venv_missing = not (PROJECT_ROOT / ".venv" / "pixelHDM-env").exists()
        old_venv_exists = (PROJECT_ROOT / ".venv" / "jit-lumina2-env").exists()

        if venv_missing and old_venv_exists:
            print("\n  To fix the venv directory issue, run:")
            print("    ren \".venv\\jit-lumina2-env\" \"pixelHDM-env\"")

        return 1


if __name__ == "__main__":
    sys.exit(main())
