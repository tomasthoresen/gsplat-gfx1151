"""Runtime GPU detection and architecture-tuned defaults.

The tuned parameters of a Gaussian splatting rasterizer are not portable
between GPU architectures. The tile size is the clearest case: it sets how many
tile-Gaussian intersections the pipeline produces, and sorting those
intersections dominates the forward pass. The value that is fastest on one
architecture is not the value that is fastest on another, because the wavefront
size, the compute unit count and the memory system all differ.

This module detects the GPU at runtime and supplies defaults chosen for it,
rather than compiling a single value in. Detection reads
``torch.cuda.get_device_properties``, so it needs no external tools and works
identically under ROCm and CUDA.

Print the detected profile with::

    python -m gsplat.device_profile

Terms:

- **Wavefront** (AMD) or **warp** (NVIDIA): the group of GPU lanes that execute
  in lockstep. RDNA parts use 32 lanes, CDNA and GCN parts 64, NVIDIA parts 32.
- **Tile size**: the edge length in pixels of the image tile one thread block
  rasterizes. A tile size of 8 gives a 64-thread block.
- **Compute unit** (AMD) or **streaming multiprocessor** (NVIDIA): the core
  the driver schedules thread blocks onto.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch

__all__ = ["DeviceProfile", "get_device_profile", "default_tile_size"]


# Tile size per architecture family.
#
# rdna:    measured on gfx1151 (Radeon 8060S, wave32, 20 CUs). At 300k
#          Gaussians on a real scene, tile 16 trains at 29.3 it/s against
#          25.4 it/s at tile 8; tile 32 is 28.3 it/s. Tile 8 produces 11.7M
#          tile-Gaussian intersections where tile 16 produces 3.3M, and the
#          radix sort over those intersections is 97% of the forward pass.
#          Applied to RDNA generally by inference from the shared wave32
#          property; only gfx1151 is measured.
# cdna:    8, the value upstream ROCm/gsplat selected for AMD Instinct, where
#          a 64-thread block is exactly one wavefront. Not measured here.
# nvidia:  16, the upstream nerfstudio gsplat default.
_TILE_SIZE_BY_FAMILY = {
    "rdna": 16,
    "cdna": 8,
    "nvidia": 16,
    "unknown": 16,
}


def _classify_amd_arch(arch: str) -> str:
    """Map a gfx target name to an architecture family."""
    # gfx900-gfx90c are GCN/Vega, gfx90a and gfx94x are CDNA; all run wave64.
    # gfx10xx, gfx11xx and gfx12xx are RDNA and run wave32.
    for prefix, family in (("gfx10", "rdna"), ("gfx11", "rdna"), ("gfx12", "rdna"),
                           ("gfx9", "cdna")):
        if arch.startswith(prefix):
            return family
    return "unknown"


@dataclass(frozen=True)
class DeviceProfile:
    """Properties of a GPU, and the rasterizer defaults chosen for it."""

    name: str
    arch: str
    family: str
    vendor: str
    wavefront: int
    compute_units: int
    unified_memory: bool
    shared_memory_per_block: int
    total_memory: int
    tile_size: int
    tile_size_source: str

    def summary(self) -> str:
        mem = self.total_memory / 1024 ** 3
        lines = [
            f"GPU              : {self.name}",
            f"Architecture     : {self.arch} ({self.family}, {self.vendor})",
            f"Wavefront size   : {self.wavefront}",
            f"Compute units    : {self.compute_units}",
            f"Shared mem/block : {self.shared_memory_per_block // 1024} KiB",
            f"Memory           : {mem:.1f} GiB"
            + (" (unified with host)" if self.unified_memory else ""),
            f"Tile size        : {self.tile_size} ({self.tile_size_source})",
        ]
        return "\n".join(lines)


@functools.lru_cache(maxsize=8)
def _profile_for_index(index: int) -> DeviceProfile:
    props = torch.cuda.get_device_properties(index)
    vendor = "amd" if torch.version.hip is not None else "nvidia"

    if vendor == "amd":
        # gcnArchName may carry target features, e.g. "gfx90a:sramecc+:xnack-"
        arch = getattr(props, "gcnArchName", "unknown").split(":")[0]
        family = _classify_amd_arch(arch)
    else:
        arch = f"sm_{props.major}{props.minor}"
        family = "nvidia"

    wavefront = getattr(props, "warp_size", 64 if family == "cdna" else 32)

    override = os.environ.get("GSPLAT_TILE_SIZE")
    if override:
        try:
            tile_size = int(override)
        except ValueError:
            raise ValueError(
                f"GSPLAT_TILE_SIZE must be an integer, got {override!r}"
            ) from None
        if tile_size <= 0 or tile_size * tile_size > props.max_threads_per_block:
            raise ValueError(
                f"GSPLAT_TILE_SIZE={tile_size} is invalid on this device: a tile "
                f"produces a block of {tile_size * tile_size} threads and the "
                f"maximum is {props.max_threads_per_block}."
            )
        source = "GSPLAT_TILE_SIZE"
    else:
        tile_size = _TILE_SIZE_BY_FAMILY.get(family, _TILE_SIZE_BY_FAMILY["unknown"])
        source = f"tuned for {family}" if family != "unknown" else "default"

    return DeviceProfile(
        name=props.name,
        arch=arch,
        family=family,
        vendor=vendor,
        wavefront=wavefront,
        compute_units=props.multi_processor_count,
        unified_memory=bool(getattr(props, "is_integrated", 0)),
        shared_memory_per_block=props.shared_memory_per_block,
        total_memory=props.total_memory,
        tile_size=tile_size,
        tile_size_source=source,
    )


def get_device_profile(
    device: Optional[Union[torch.device, str, int]] = None
) -> DeviceProfile:
    """Return the profile of a CUDA/HIP device.

    Args:
        device: the device to describe. Defaults to the current device.

    Raises:
        RuntimeError: if no GPU is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/HIP device is available.")
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
    return _profile_for_index(index)


def default_tile_size(
    device: Optional[Union[torch.device, str, int]] = None
) -> int:
    """Return the tile size chosen for the device's architecture.

    Falls back to 16 when no GPU is present, so that importing and calling into
    the library on a CPU-only host does not raise.
    """
    try:
        return get_device_profile(device).tile_size
    except Exception:
        return _TILE_SIZE_BY_FAMILY["unknown"]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/HIP device is available.")
    for i in range(torch.cuda.device_count()):
        if torch.cuda.device_count() > 1:
            print(f"--- device {i} ---")
        print(_profile_for_index(i).summary())
