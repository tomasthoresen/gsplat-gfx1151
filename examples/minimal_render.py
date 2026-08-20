#!/usr/bin/env python3
"""Minimal gsplat example: render 3D Gaussians, then fit new ones to that render.

No dataset and no COLMAP reconstruction are needed. The script builds a small
scene of coloured Gaussians in code, renders it from a fixed camera, and then
optimises a second, randomly initialised set of Gaussians until its render
matches the first. The first half exercises the forward rasterizer, the second
half exercises the backward pass.

Run:
    python examples/minimal_render.py --out-dir renders

Writes target.png (the scene) and fitted.png (the optimisation result).
"""

import argparse
import math
import os

import torch
from PIL import Image

from gsplat import rasterization


def make_camera(width, height, distance, device):
    """A camera looking down +Z from `distance` away, with a 90 degree field of view."""
    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, distance],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    focal = 0.5 * width / math.tan(0.5 * math.pi / 2)
    K = torch.tensor(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        device=device,
    )
    return viewmat[None], K[None]


def render(means, quats, scales, opacities, colors, viewmats, Ks, width, height):
    """Rasterize Gaussians to an [H, W, 3] image in [0, 1].

    gsplat expects normalised quaternions, positive scales, and opacities and
    colours in [0, 1]; the raw parameters are passed through the corresponding
    activations here so they can be optimised unconstrained.
    """
    renders, _alphas, _info = rasterization(
        means,
        quats / quats.norm(dim=-1, keepdim=True),
        scales.exp(),
        opacities.sigmoid(),
        colors.sigmoid(),
        viewmats,
        Ks,
        width,
        height,
        packed=False,
    )
    return renders[0]


def build_target_scene(device, n_rings=3, per_ring=12):
    """A ring arrangement of coloured Gaussians, deterministic."""
    torch.manual_seed(0)
    means, colors = [], []
    for ring in range(n_rings):
        radius = 0.4 + 0.45 * ring
        for i in range(per_ring):
            angle = 2 * math.pi * i / per_ring
            means.append([radius * math.cos(angle), radius * math.sin(angle), 0.0])
            hue = i / per_ring
            colors.append(
                [
                    0.5 + 0.5 * math.cos(2 * math.pi * (hue + 0.00)),
                    0.5 + 0.5 * math.cos(2 * math.pi * (hue + 0.33)),
                    0.5 + 0.5 * math.cos(2 * math.pi * (hue + 0.67)),
                ]
            )
    n = len(means)
    means = torch.tensor(means, device=device)
    colors = torch.tensor(colors, device=device)

    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), math.log(0.10), device=device)
    opacities = torch.full((n,), 4.0, device=device)  # sigmoid(4) ~ 0.98
    # colors are stored pre-sigmoid, so invert the activation
    colors = torch.logit(colors.clamp(0.01, 0.99))
    return means, quats, scales, opacities, colors


def save_png(image, path):
    array = (image.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="renders")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-gaussians", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. Check the ROCm install.")

    device = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(0)}")
    os.makedirs(args.out_dir, exist_ok=True)
    width = height = args.size
    viewmats, Ks = make_camera(width, height, distance=4.0, device=device)

    # --- forward: render the target scene -------------------------------
    print("\nrendering target scene")
    target_params = build_target_scene(device)
    with torch.no_grad():
        target = render(*target_params, viewmats, Ks, width, height)
    print(f"  image {tuple(target.shape)}, range "
          f"[{target.min():.3f}, {target.max():.3f}]")
    save_png(target, os.path.join(args.out_dir, "target.png"))

    # --- backward: fit fresh Gaussians to that render --------------------
    print(f"\nfitting {args.num_gaussians} Gaussians over {args.steps} steps")
    torch.manual_seed(1)
    n = args.num_gaussians
    means = (torch.rand(n, 3, device=device) * 2 - 1).requires_grad_(True)
    quats = torch.randn(n, 4, device=device).requires_grad_(True)
    scales = torch.full((n, 3), math.log(0.1), device=device).requires_grad_(True)
    opacities = torch.zeros(n, device=device).requires_grad_(True)
    colors = torch.zeros(n, 3, device=device).requires_grad_(True)
    params = [means, quats, scales, opacities, colors]

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3},
            {"params": [scales], "lr": 5e-3},
            {"params": [opacities], "lr": 5e-2},
            {"params": [colors], "lr": 2e-2},
        ]
    )

    first_loss = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        image = render(*params, viewmats, Ks, width, height)
        loss = (image - target).abs().mean()
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  L1 {loss.item():.5f}")

    print(f"\n  L1 {first_loss:.5f} -> {loss.item():.5f}")
    with torch.no_grad():
        save_png(render(*params, viewmats, Ks, width, height),
                 os.path.join(args.out_dir, "fitted.png"))
    print("\ndone")


if __name__ == "__main__":
    main()
