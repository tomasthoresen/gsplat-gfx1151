"""Wave-size correctness tests for the ROCm backward rasterization kernels.

The backward kernels reduce a value across a warp and then have a single lane
write the result out with an atomic. That is only correct when the warp tile
width, the reduction width, and the hardware wavefront all agree. When they do
not -- for example ``cg::tiled_partition<64>`` combined with a 32-lane rocPRIM
reduction on a wave32 part -- entire waves' partial sums are silently dropped
and the gradients come out scaled by a constant factor. Nothing crashes and
training still converges, so a smoke test that only checks "loss goes down"
does not catch it.

These tests validate the backward pass by finite differences against the
forward pass. The forward kernels contain no warp-collective reductions and no
block-size specialisation, so they are the trustworthy side of the comparison.
No reference implementation is required, which keeps the tests runnable on
ROCm without nerfacc.

``tile_size`` is parametrised because ``tile_size * tile_size == 64`` selects a
separate block-size-64 kernel on ROCm that assumes the block is exactly one
wavefront.

Usage:
```bash
pytest tests/test_wave_backward.py -s
```
"""

import pytest
import torch

device = torch.device("cuda:0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA/ROCm device"
)


def _scene(n_gaussians=150, width=64, height=64, channels=3, seed=0):
    """A small deterministic scene, projected ready for rasterization."""
    from gsplat.cuda._wrapper import fully_fused_projection, quat_scale_to_covar_preci

    torch.manual_seed(seed)
    means = torch.randn(n_gaussians, 3, device=device) * 0.4 + torch.tensor(
        [0.0, 0.0, 3.0], device=device
    )
    quats = torch.randn(n_gaussians, 4, device=device)
    scales = torch.rand(n_gaussians, 3, device=device) * 0.08 + 0.03
    opacities = (torch.rand(n_gaussians, device=device) * 0.6 + 0.2)[None]
    colors = torch.rand(n_gaussians, channels, device=device)[None]

    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[[60.0, 0.0, width / 2], [0.0, 60.0, height / 2], [0.0, 0.0, 1.0]]],
        device=device,
    )
    covars, _ = quat_scale_to_covar_preci(
        quats, scales, compute_preci=False, triu=True
    )
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )
    return dict(
        means2d=means2d,
        conics=conics,
        colors=colors,
        opacities=opacities,
        radii=radii,
        depths=depths,
        width=width,
        height=height,
    )


def _isect(scene, tile_size):
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    tw = (scene["width"] + tile_size - 1) // tile_size
    th = (scene["height"] + tile_size - 1) // tile_size
    _, isect_ids, flatten_ids = isect_tiles(
        scene["means2d"], scene["radii"], scene["depths"], tile_size, tw, th
    )
    return isect_offset_encode(isect_ids, 1, tw, th), flatten_ids


def _render_sum(scene, tile_size, isect_offsets, flatten_ids, colors, opacities):
    """Scalar objective: the sum of all rendered pixels."""
    from gsplat.cuda._wrapper import rasterize_to_pixels

    renders, _ = rasterize_to_pixels(
        scene["means2d"],
        scene["conics"],
        colors,
        opacities,
        scene["width"],
        scene["height"],
        tile_size,
        isect_offsets,
        flatten_ids,
    )
    return renders.double().sum()


def _analytic_grads(scene, tile_size, isect_offsets, flatten_ids):
    colors = scene["colors"].detach().clone().requires_grad_(True)
    opacities = scene["opacities"].detach().clone().requires_grad_(True)
    _render_sum(
        scene, tile_size, isect_offsets, flatten_ids, colors, opacities
    ).backward()
    return colors.grad.clone(), opacities.grad.clone()


@pytest.mark.parametrize("tile_size", [8, 16])
@pytest.mark.parametrize("channels", [3, 32])
@pytest.mark.parametrize("param", ["colors", "opacities"])
def test_backward_matches_finite_differences(tile_size, channels, param):
    """Analytic gradients must match central differences through the forward pass.

    A wave-size mismatch shows up here as a near-constant ratio between the
    analytic and numeric gradient (0.5 when half the waves are dropped).
    """
    scene = _scene(channels=channels)
    isect_offsets, flatten_ids = _isect(scene, tile_size)
    g_colors, g_opacities = _analytic_grads(
        scene, tile_size, isect_offsets, flatten_ids
    )

    tensor = scene[param].detach().clone()
    grad = g_colors if param == "colors" else g_opacities
    eps = 2e-4

    # Check the entries with the largest gradient: those carry the most signal
    # and are least affected by finite-difference noise.
    indices = grad.abs().flatten().topk(5).indices.tolist()
    ratios = []
    for i in indices:
        perturbed = tensor.clone()
        flat = perturbed.view(-1)
        original = flat[i].item()

        def render_with(value):
            flat[i] = value
            colors = perturbed if param == "colors" else scene["colors"]
            opacities = perturbed if param == "opacities" else scene["opacities"]
            return _render_sum(
                scene, tile_size, isect_offsets, flatten_ids, colors, opacities
            ).item()

        numeric = (render_with(original + eps) - render_with(original - eps)) / (2 * eps)
        analytic = grad.view(-1)[i].item()
        assert abs(numeric) > 1e-6, "degenerate finite difference; adjust the scene"
        ratios.append(analytic / numeric)

    worst = max(abs(r - 1.0) for r in ratios)
    assert worst < 5e-3, (
        f"{param} gradient disagrees with finite differences at tile_size="
        f"{tile_size}, channels={channels}: analytic/numeric ratios {ratios}. "
        "A near-constant ratio indicates dropped warp partial sums (check that "
        "the warp tile width matches GSPLAT_WARP_SIZE)."
    )


@pytest.mark.parametrize("channels", [3, 32])
def test_gradients_are_tile_size_invariant(channels):
    """Tile size is an implementation detail; gradients must not depend on it.

    On ROCm, tile_size=8 gives a 64-thread block and selects a distinct
    block-size-64 backward kernel, so this compares two different kernels.
    """
    scene = _scene(channels=channels)
    grads = {}
    for tile_size in (8, 16):
        isect_offsets, flatten_ids = _isect(scene, tile_size)
        grads[tile_size] = _analytic_grads(
            scene, tile_size, isect_offsets, flatten_ids
        )

    for name, a, b in zip(("colors", "opacities"), grads[8], grads[16]):
        torch.testing.assert_close(
            a,
            b,
            rtol=2e-3,
            atol=2e-3,
            msg=lambda s, name=name: f"{name} gradients differ between "
            f"tile_size=8 and tile_size=16:\n{s}",
        )


def test_forward_is_tile_size_invariant():
    """Guards the assumption that the forward pass is the trustworthy side."""
    scene = _scene()
    values = []
    for tile_size in (8, 16):
        isect_offsets, flatten_ids = _isect(scene, tile_size)
        values.append(
            _render_sum(
                scene,
                tile_size,
                isect_offsets,
                flatten_ids,
                scene["colors"],
                scene["opacities"],
            ).item()
        )
    assert abs(values[0] - values[1]) / abs(values[1]) < 1e-6, (
        f"forward render differs between tile sizes: {values}"
    )
