"""Tests for runtime GPU detection and the architecture-tuned defaults.

The tuned value that matters is the tile size: it sets how many tile-Gaussian
intersections the pipeline produces, and sorting those dominates the forward
pass. These tests check that detection reports the real device, that the tuned
value is consistent with the detected architecture, that an explicit tile size
still wins, and that selecting a tile size never changes what is rendered.
"""

import os

import pytest
import torch

from gsplat.device_profile import (
    DeviceProfile,
    _classify_amd_arch,
    _profile_for_index,
    default_tile_size,
    get_device_profile,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA/ROCm device"
)


@pytest.mark.parametrize(
    "arch,family",
    [
        ("gfx1151", "rdna"),   # RDNA3.5, Strix Halo
        ("gfx1100", "rdna"),   # RDNA3, Radeon 7900
        ("gfx1030", "rdna"),   # RDNA2
        ("gfx1200", "rdna"),   # RDNA4
        ("gfx942", "cdna"),    # CDNA3, MI300
        ("gfx90a", "cdna"),    # CDNA2, MI200
        ("gfx906", "cdna"),    # GCN/Vega, also wave64
        ("nonsense", "unknown"),
    ],
)
def test_amd_arch_classification(arch, family):
    assert _classify_amd_arch(arch) == family


def test_profile_describes_the_real_device():
    p = get_device_profile()
    assert isinstance(p, DeviceProfile)
    assert p.wavefront in (32, 64)
    assert p.compute_units > 0
    assert p.total_memory > 0
    assert p.tile_size > 0
    # A tile must fit in a thread block.
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert p.tile_size ** 2 <= props.max_threads_per_block
    assert p.summary()


def test_tuned_tile_size_matches_detected_family():
    """RDNA runs wave32 and is tuned to 16; CDNA runs wave64 and is tuned to 8."""
    p = get_device_profile()
    if p.tile_size_source.startswith("GSPLAT_TILE_SIZE"):
        pytest.skip("tile size is overridden by the environment")
    if p.family == "rdna":
        assert p.wavefront == 32
        assert p.tile_size == 16
    elif p.family == "cdna":
        assert p.wavefront == 64
        assert p.tile_size == 8


def test_profile_is_cached():
    assert get_device_profile() is get_device_profile()


def test_env_override(monkeypatch):
    monkeypatch.setenv("GSPLAT_TILE_SIZE", "8")
    _profile_for_index.cache_clear()
    try:
        p = _profile_for_index(torch.cuda.current_device())
        assert p.tile_size == 8
        assert p.tile_size_source == "GSPLAT_TILE_SIZE"
    finally:
        _profile_for_index.cache_clear()


@pytest.mark.parametrize("bad", ["0", "-4", "64", "not-a-number"])
def test_env_override_rejects_invalid(monkeypatch, bad):
    """A tile of 64 gives a 4096-thread block, above every current limit."""
    monkeypatch.setenv("GSPLAT_TILE_SIZE", bad)
    _profile_for_index.cache_clear()
    try:
        with pytest.raises(ValueError):
            _profile_for_index(torch.cuda.current_device())
    finally:
        _profile_for_index.cache_clear()


def test_default_tile_size_agrees_with_profile():
    assert default_tile_size() == get_device_profile().tile_size


def _tiny_scene(device, n=400):
    torch.manual_seed(0)
    means = torch.randn(n, 3, device=device) * 0.4 + torch.tensor(
        [0.0, 0.0, 3.0], device=device
    )
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(n, 3, device=device) * 0.05 + 0.02
    opacities = torch.rand(n, device=device) * 0.6 + 0.3
    colors = torch.rand(n, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]]], device=device
    )
    return means, quats, scales, opacities, colors, viewmats, Ks


def test_auto_tile_size_matches_explicit():
    """Passing None must render exactly what passing the tuned value renders."""
    from gsplat import rasterization

    device = torch.device("cuda:0")
    args = _tiny_scene(device)
    auto, _, _ = rasterization(*args, 64, 64, tile_size=None)
    explicit, _, _ = rasterization(*args, 64, 64, tile_size=default_tile_size())
    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)


def test_explicit_tile_size_is_respected():
    """An explicit value must not be replaced by the tuned one."""
    from gsplat import rasterization

    device = torch.device("cuda:0")
    args = _tiny_scene(device)
    out, _, meta = rasterization(*args, 64, 64, tile_size=8)
    assert meta["tile_size"] == 8
    assert torch.isfinite(out).all()
