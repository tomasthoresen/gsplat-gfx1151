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
held-out views, identical seed cloud (600k points).

**Provenance** — gfx1151 (Radeon 8060S, Strix Halo), **wave32**, ROCm 7.2.1,
PyTorch 2.13.0+rocm7.2, Python 3.12, venv `/home/amd/imgen`, this branch at
`75baa7b`, upstream pin `b01acd4`. Reference: NVIDIA RTX A4000, upstream gsplat
1.5.3, torch 2.5.1+cu121, conda env `sam3d-objects`.

| build | held-out PSNR | gaussians | wall clock |
|---|---|---|---|
| wave32 reductions only | 22.25 dB | 491,565 | 4.7 min |
| + cooperative-group tiles | 22.94 dB | 591,724 | 5.1 min |
| **+ projection lane-id (this branch)** | **23.67 dB** | **608,816** | 5.2 min |
| NVIDIA RTX A4000 (upstream gsplat) | 23.64 dB | 607,821 | 3.8 min |

**Parity with a discrete A4000 at 1.37× the wall clock**, on an integrated GPU.
The 0.03 dB and 0.16% Gaussian difference is float noise — see *Validation*.

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

### 4. Projection lane id and lane-scan loops

The one that mattered most, and the one a training comparison cannot find.

```c
- unsigned int warp_thread_id = threadIdx.x % 64;
+ unsigned int warp_thread_id = threadIdx.x % 32;
- for (int i = 0; i < 64; ++i) {        // lane scan for the leader election
+ for (int i = 0; i < 32; ++i) {
```

in all four `Projection*.cu` kernels and in `manual_dynamic_reduce_sum_*` in
`Utils.cuh`.

These kernels reduce per-Gaussian gradients across the lanes that share a
Gaussian id, elect a leader lane, and have only the leader do the atomic write.
On wave32, `threadIdx.x % 64` gives half the threads a lane id in 32–63, which
can never equal a leader lane id in 0–31 — so **those threads never win the
election and their gradients are silently dropped**. Exactly half.

It affected only `v_means`, `v_scales`, `v_quats` and `v_covar`, because those
are the projection outputs. Opacity and colour gradients come from the
rasteriser and were always correct, which is what made the signature legible.

Note these kernels already used `cg::tiled_partition<32>` — the tile was right
and the lane arithmetic around it was not.

## Validation

Training comparisons confound kernel numerics with densification decisions and
view sampling. `kernel_probe.py` removes all three: identical tensors from a
fixed CPU seed, one forward, one backward, results dumped raw and diffed
against the same script on the A4000.

Forward agrees to float precision from the start — `img_mean` identical to ten
significant figures, loss differing at 1.2e-07 (reduction order).

Backward, before and after the lane-id fix:

| tensor | before | after | NVIDIA |
|---|---|---|---|
| `g_means` non-zero | 18,222 | 36,282 | 36,282 |
| `g_scales` non-zero | 18,222 | 36,282 | 36,282 |
| `g_opac` non-zero | 12,094 | 12,094 | 12,094 |
| `g_colors` non-zero | 36,282 | 36,282 | 36,282 |
| `g_means` Σ\|g\| ratio | 0.523 | 1.00000 | 1 |
| `g_means` mean rel err | 4.98e-01 | 3.06e-05 | — |

Half the gradient entries missing, in exactly the two tensors the projection
kernels write. After the fix every tensor matches in count and magnitude, with
relative error at float32 noise.

**Reproduce:** `python kernel_probe.py out.npz` on each machine, then diff the
arrays. Do this before trusting any wave-size port — a training run that scores
1.4 dB low looks like tuning, not a dropped half of the gradient.
