"""
Strict Test Suite

Critical tests that verify core functionality with strict validation.

These tests are designed to:
1. Not trust past documentation - verify directly from source code
2. Apply rigorous mathematical validation
3. Test edge cases and boundary conditions
4. Prevent regression of previously fixed bugs
5. Validate end-to-end configuration to execution paths

Test modules:
    - test_gradient_flow_strict: Verify gradient flow through the model (CRITICAL)
    - test_flow_matching_strict: Verify Flow Matching math (CRITICAL)
    - test_config_strict: Verify configuration parsing (CRITICAL)
    - test_losses_strict: Verify loss functions (HIGH)
    - test_trainer_strict: Verify training logic (HIGH)
    - test_sampler_strict: Verify sampling/inference (HIGH)
    - test_data_e2e_strict: Verify data system E2E (MEDIUM)
    - test_pixeldit_io_strict: Verify model I/O formats (MEDIUM)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2026-01-02
"""
