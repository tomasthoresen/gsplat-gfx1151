# gsplat on gfx1151 (Strix Halo / Radeon 8060S)

A wave32 port of [ROCm/gsplat](https://github.com/ROCm/gsplat), which targets
AMD Instinct MI300X. MI300X is CDNA with 64-wide wavefronts; RDNA 3.5 is 32-wide,
and every warp-level primitive in the fork is sized for 64.

Upstream is kept as the `upstream` remote, so this branch is a reviewable diff:

```bash
git diff upstream/main..gfx1151
```

## Result

Same 240-view capture of a 60 × 40 m office interior, 5000 iterations, 30
held-out views, identical seed cloud (600k points):

| build | held-out PSNR | gaussians | wall clock |
|---|---|---|---|
| AMD gfx1151, cg tiles left at 64 | 22.25 dB | 491,565 | 4.7 min |
| **AMD gfx1151, this branch** | **22.94 dB** | **591,724** | 5.1 min |
| NVIDIA RTX A4000 (upstream gsplat) | 23.64 dB | 607,821 | 3.8 min |

An integrated GPU within 0.70 dB of a discrete A4000 at 1.34× the wall clock.
The residual gap is unexplained — see *Known gap* below.

## Environment

- gfx1151 (Radeon 8060S, Strix Halo), 62 GB unified memory
- ROCm 7.2.1, PyTorch 2.13.0+rocm7.2, Python 3.12
- Ubuntu, `hipcc` from `/opt/rocm`

## Build

```bash
git submodule update --init --depth 1 --recursive   # glm is a submodule
GLM=$PWD/gsplat/cuda/csrc/third_party/glm

PYTORCH_ROCM_ARCH=gfx1151 GPU_ARCHS=gfx1151 MAX_JOBS=8 \
CPATH="$GLM" \
CXXFLAGS="-D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=0 -DGLM_FORCE_PURE" \
python -m pip install --no-build-isolation -e .
```

`CPATH` rather than installing glm to `/usr/local/include`: it needs no sudo and
survives hipify regenerating `gsplat/hip/` on every build, which is what breaks
the vendored relative include path.

`setup.py` detects the arch from `rocminfo` and gets gfx1151 right on real
hardware. The gfx942 fallback that other write-ups patch out only bites inside
Docker, where there is no GPU to query.

## What changed, and why

### 1. Wavefront width: 64 → 32

`Utils.cuh` template defaults, every explicit call-site template argument
(`rocprim_warpSum<64>`, `<CDIM, 64>`, `<3, 64>`), and the matching
`rocprim::warp_reduce<T, 64>::storage_type` instantiations.

**A global `sed 's/64/32/g'` is not safe here.** Two reasons:

- The shared-memory allocation is sized `warps_per_block = (block_size + 63) / 64`
  while the index into it is `threadIdx.x / LOGICAL_WARP_SIZE`. Halve the warp
  size without growing the allocation and the kernel overruns shared memory —
  silently, with corrupted gradients rather than a crash. The count is patched
  to `(block_size + 31) / 32` in all three backward kernels.
- `if (block_size == 64)` in `RasterizeToPixels3DGSBwd.cu` is a **tile-size
  dispatch** (`block_size = tile_size * tile_size`), not a wavefront constant. A
  global sed rewrites it and changes which kernel variant runs. Left alone.

### 2. Cooperative-group tiles

```c
#if USE_ROCM
cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);   // CDNA
#else
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
#endif
```

A tile wider than the hardware wavefront, with `__shfl` reads inside it
addressing lanes that do not exist. This was worth **+0.69 dB and 100k
Gaussians** on its own — the single largest correctness fix in the port.

The DPP reduction paths (`dpp_sclr_warpSum`, `dpp_sprd_warpSum`) needed nothing:
their `__builtin_amdgcn_mov_dpp` sequences are commented out upstream and
delegate to `rocprim_warpSum`, so they follow the change above.

### 3. PyTorch 2.13 allocator namespace

```c
- auto &caching_allocator = *::c10::hip::HIPCachingAllocator::get();
+ auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();
```

Not a wavefront issue — a torch version break. `c10/hip/HIPCachingAllocator.h`
in 2.13 declares `namespace c10::cuda::CUDACachingAllocator`; the
`c10::hip::HIPCachingAllocator` alias the fork was written against (torch
2.6–2.9) is gone. Fails at compile time with
`no member named 'HIPCachingAllocator' in namespace 'c10::hip'`.

## Known gap

The remaining 0.70 dB and 3% Gaussian deficit against the A4000 is not
explained. Reduction order differs between architectures, so bitwise-identical
results were never expected, but a *systematic* Gaussian deficit points at
gradients coming out slightly small rather than at float noise — densification
in `DefaultStrategy` is thresholded on accumulated gradient magnitude.

Unexcluded candidates: `manual_warpSum`, and `rocprim::warp_reduce` behaviour
when the logical warp size equals the hardware wavefront versus when it does not.

Treat upstream on NVIDIA as the reference for published numbers. This branch is
for iteration, and for the thing it uniquely enables: real-time splat rendering
on the same machine as the simulator.

## Validation

`rasterization()` forward and backward both produce finite, non-zero gradients,
and training loss at iteration 0 matches the A4000 exactly (0.3923), confirming
the forward path agrees at initialisation.
