import torch

from src.config import PixelHDMConfig
from src.training.flow_matching import PixelHDMFlowMatching

from tests.conftest import assert_gradient_flow


def test_flow_matching_gradients_with_dts():
    config = PixelHDMConfig.for_testing()
    config.use_dynamic_timestep_shift = True
    config.timestep_shift_type = "linear"
    config.timestep_shift_base_shift = 0.5
    config.timestep_shift_max_shift = 1.15
    config.timestep_shift_base_seq_len = 16
    config.timestep_shift_max_seq_len = 256
    config.timestep_shift_clamp_linear = True

    flow = PixelHDMFlowMatching(config=config)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=1, bias=True)

        def forward(self, x, t):
            return self.conv(x)

    model = DummyModel()
    images = torch.randn(2, 3, 64, 64)

    t, z_t, x_clean, noise = flow.prepare_training(images)
    v_pred = model(z_t, t)
    loss = flow.compute_loss(v_pred=v_pred, x=x_clean, noise=noise)

    assert_gradient_flow(model, loss)
