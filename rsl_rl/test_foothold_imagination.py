import math

import torch

from rsl_rl.modules.foothold_imagination import (
    FootholdImagination,
    support_deficiency_from_heights,
    world_xy_to_base_yaw,
)
from rsl_rl.storage.foothold_replay_buffer import FootholdReplayBuffer


def test_world_xy_to_base_yaw_rotates_and_translates():
    target = torch.tensor([[2.0, 3.0]])
    base = torch.tensor([[1.0, 1.0]])
    yaw = torch.tensor([math.pi / 2.0])
    actual = world_xy_to_base_yaw(target, base, yaw)
    torch.testing.assert_close(actual, torch.tensor([[2.0, -1.0]]), atol=1e-6, rtol=0.0)


def test_gaussian_nll_is_finite_and_masked():
    model = FootholdImagination(8, 3, hidden_dims=(16, 8))
    privileged_obs = torch.randn(4, 8)
    actions = torch.randn(4, 3)
    mu, sigma = model(privileged_obs, actions)
    targets = torch.randn(4, 2, 2)
    valid = torch.tensor([[True, False], [True, True], [False, False], [True, False]])
    loss = model.gaussian_nll(mu, sigma, targets, valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_support_deficiency_counts_missing_and_low_samples():
    heights = torch.tensor([[1.0, 0.99, 0.8, float("inf")]])
    deficiency = support_deficiency_from_heights(
        heights, torch.tensor([[1.0]]), tolerance=0.03
    )
    torch.testing.assert_close(deficiency, torch.tensor([0.5]))


def test_foothold_replay_buffer_wraps():
    replay = FootholdReplayBuffer(4, 2, buffer_size=5, device="cpu")
    replay.insert(
        torch.arange(28, dtype=torch.float).view(7, 4),
        torch.arange(14, dtype=torch.float).view(7, 2),
        torch.arange(14, dtype=torch.float).view(7, 2),
        torch.arange(7) % 2,
        torch.arange(7) % 2,
    )
    assert replay.num_samples == 5
    sample = replay.sample(3)
    assert [item.shape for item in sample] == [(3, 4), (3, 2), (3, 2), (3,), (3,)]
