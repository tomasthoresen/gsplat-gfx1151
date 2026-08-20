# Porting log — gsplat on ROCm/gfx1151

Author: Tomas Thoresen <tomasthoresen@gmail.com>

Append-only record of every kernel or file changed during the port, the
CUDA→HIP substitution made, and the reason. One entry per change, newest last.

Entry format:

```
## YYYY-MM-DD — <repo>:<file>
- Change: <what was substituted or rewritten>
- Reason: <why the AMD backend requires it>
- Verification: <test or parity check that covers it>
- Commit: <hash in the target repo>
```

---

## 2026-08-04 — baseline (no source changes)

- Upstream pin: ROCm/gsplat
  `b01acd43e3c7fa942f95fda0974e9125e4de7395` (branch `main`), vendored glm submodule
  `33b4a621a697a305bc3a7610d290677b96beb181`.
- Host: ROCm 7.2.1 (`/opt/rocm → /opt/rocm-7.2.1`), PyTorch 2.13.0+rocm7.2
  (HIP 7.2.53211), Python 3.12, venv `/home/amd/imgen`.
- Target: gfx1151 (Radeon 8060S, Strix Halo), **wave32**, 62 GB unified memory.
- Reference for parity: NVIDIA RTX A4000, upstream gsplat 1.5.3,
  torch 2.5.1+cu121, conda env `sam3d-objects` on `tomas@192.168.50.8`.
- Upstream targets AMD Instinct MI300X — CDNA, **wave64**. Every warp-level
  primitive in the fork is sized for 64 lanes. That is the entire port surface.

## 2026-08-04 — gsplat/cuda/include/Utils.cuh
- Change: `LOGICAL_WARP_SIZE` template defaults 64 → 32; lane-scan loops in
  `manual_dynamic_reduce_sum_{vec3,vec4,mat3}` bounded 64 → 32.
- Reason: RDNA 3.5 wavefronts are 32 lanes. A rocprim logical warp wider than
  the hardware wavefront, and a scan over lanes 32–63, address lanes that do
  not exist.
- Verification: `kernel_probe.py` gradient parity vs A4000 golden output.
- Commit: 9bf6896, 75baa7b

## 2026-08-04 — gsplat/cuda/csrc/RasterizeToPixels{2DGS,3DGS,FromWorld3DGS}Bwd.cu
- Change: `rocprim_warpSum<64>` / `<CDIM,64>` / `<3,64>` → 32; matching
  `rocprim::warp_reduce<T,64>::storage_type` → 32; shared-memory allocation
  count `(block_size + 63) / 64` → `(block_size + 31) / 32`;
  `cg::tiled_partition<64>` → `<32>` under `#if USE_ROCM`.
- Reason: wave32. **A global `sed 's/64/32/g'` is unsafe here** — the shared
  memory is sized from the warp count while indexed by
  `threadIdx.x / LOGICAL_WARP_SIZE`, so halving the warp size without growing
  the allocation overruns shared memory silently; and `if (block_size == 64)`
  is a tile-size dispatch, not a wavefront constant, and must be left alone.
- Verification: held-out PSNR 22.25 → 22.94 dB on the reference capture.
- Commit: 9bf6896

## 2026-08-04 — gsplat/cuda/include/Common.cuh
- Change: `::c10::hip::HIPCachingAllocator::get()` →
  `::c10::cuda::CUDACachingAllocator::get()` in the ROCm `CUB_WRAPPER` branch.
- Reason: **not a wavefront issue** — a torch version break. torch 2.13 declares
  `namespace c10::cuda::CUDACachingAllocator` inside
  `c10/hip/HIPCachingAllocator.h`; the `c10::hip::` alias this fork was written
  against (torch 2.6–2.9) is gone.
- Verification: compiles; upstream fails with
  `no member named 'HIPCachingAllocator' in namespace 'c10::hip'`.
- Commit: 9bf6896

## 2026-08-04 — gsplat/cuda/csrc/Projection{EWA3DGS,2DGS}{Fused,Packed}.cu
- Change: `warp_thread_id = threadIdx.x % 64` → `% 32`; leader-election
  lane-scan loops `i < 64` → `i < 32`.
- Reason: these kernels reduce per-gaussian gradients across the lanes sharing a
  gaussian id, elect a leader lane, and let only the leader do the atomic write.
  On wave32, `% 64` gives half the threads a lane id in 32–63, which can never
  equal a leader lane id in 0–31 — so those threads never win the election and
  **their gradients are never written**. Exactly half, silently, no error.
  Affected only `v_means`, `v_scales`, `v_quats`, `v_covar` (projection
  outputs); opacity and colour come from the rasteriser and were always correct.
  Note the tile was already `cg::tiled_partition<32>` — the tile was right and
  the lane arithmetic wrapped around it was not.
- Verification: `kernel_probe.py` — `g_means`/`g_scales` non-zero count
  18,222 → 36,282 (matching A4000 exactly), Σ|g| ratio 0.523 → 1.00000, mean
  relative error 4.98e-01 → 3.06e-05. Held-out PSNR 22.94 → 23.67 dB vs
  23.64 dB on the A4000.
- Commit: 75baa7b

## 2026-08-20 — gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu

- Change: the `block_size == 64` dispatch selecting
  `rasterize_bs64_to_pixels_3dgs_bwd_kernel` is gated on
  `GSPLAT_WAVE_SIZE == 64`. The condition governs both the kernel selection and
  the shared-memory branch above it, which compute different layouts and must
  agree. On wave32 the generic kernel is used at every tile size.
- Reason: this refines the 2026-08-04 entry, which recorded that
  `if (block_size == 64)` is a tile-size dispatch rather than a wavefront
  constant and was left alone. The condition is indeed a tile-size dispatch.
  The kernel it selects is the wavefront constraint: it uses the block thread
  rank as a lane id, shuffles across 64 lanes, indexes with `e % 64` and
  `e / 64`, and its shared-memory branch sizes the allocation for a single
  wave. On wave32 a 64-thread block is two waves against a one-wave layout.
  `gsplat/rendering.py` defaults `tile_size` to 8, giving a 64-thread block, so
  this was the default path.
- Verification: with the kernel reachable, a forward and backward pass at
  `tile_size=8` aborts with `hipErrorIllegalAddress`, while `tile_size=16`
  completes. After gating, both tile sizes complete and return an identical
  colour-gradient sum of 1912.6282, restoring tile-size invariance.
  `tests/test_wave_backward.py` passes 11 of 11; against the ungated build the
  suite cannot run to completion. 500 steps of `simple_trainer.py` at the
  default tile size complete without fault.
- Commit: 4884d08

## 2026-08-20 — gsplat/cuda/include/Common.cuh

- Change: added the `GSPLAT_WAVE_SIZE` macro, default 32, naming the wavefront
  size the fork targets. Replaced `assert(res == hipSuccess)` in the ROCm
  `CUB_WRAPPER` branch with `TORCH_CHECK`.
- Reason: the wave32 constants introduced on 2026-08-04 were bare literals with
  no named invariant for new code to test against; the bs64 gate needs one. The
  C preprocessor removes `assert` under `NDEBUG`, so release builds discarded
  every rocPRIM radix-sort status.
- Verification: compiles; `tests/test_wave_backward.py` passes 11 of 11.
- Commit: 4884d08

## 2026-08-20 — setup.py

- Change: the ROCm branch stages a complete glm header tree to
  `${XDG_CACHE_HOME:-~/.cache}/gsplat-gfx1151/glm_ext` and puts it first on the
  include path. `GSPLAT_GLM_DIR` overrides the location.
- Reason: the repository did not build from a clean checkout. Upstream's ROCm
  branch supplies no glm include directory at all, unlike its CUDA branch, so
  the build depended on glm being present on a system include path; none of
  `~/.local/include`, `/opt/conda/include` or `/opt/rocm/include` carries it.
  A clean clone fails with `fatal error: glm/gtc/type_ptr.hpp: No such file or
  directory`. Staging must be outside the project tree: hipify rewrites a
  staging directory placed inside it and emits duplicate `*_hip.h` headers that
  collide with the originals.
- Verification: a clean clone with submodules initialised builds under
  PyTorch 2.11.0+rocm7.2, where it previously failed.
- Commit: 4884d08

## 2026-08-20 — gsplat/exporter.py

- Change: the `.ply` export validity check tests `torch.isnan(opacities)` and
  `torch.isinf(opacities)` per Gaussian instead of reducing the opacity array
  with `.any(dim=0)`.
- Reason: **not a wavefront issue.** The reduction collapsed the opacity array
  to a single boolean and applied it to every Gaussian, so one non-finite
  opacity discarded the entire scene on export.
- Verification: `--save_ply` writes a point cloud whose Gaussian count matches
  the trained model.
- Commit: 4884d08

## 2026-08-20 — examples/simple_trainer.py

- Change: `fused_ssim` is imported inside a `try`/`except ImportError`, and the
  structural-similarity loss term falls back to the `torchmetrics` functional
  implementation when the extension is absent.
- Reason: **not a wavefront issue.** `fused_ssim` is a separate CUDA extension
  that is not built in a ROCm environment, and the unconditional import aborted
  the trainer at startup.
- Verification: 500 training steps complete with the extension absent.
- Commit: 4884d08

## 2026-08-20 — gsplat/cuda/_backend.py

- Change: when a compiled extension exists on disk but fails to load, the
  underlying `ImportError` is raised with a rebuild instruction.
- Reason: **not a wavefront issue.** The previous branch treated any
  `ImportError` as an absent build, printed "No CUDA toolkit found", and left
  the module handle as `None`, so the real cause — commonly an ABI mismatch
  after a PyTorch upgrade — surfaced later as `AttributeError` on `NoneType`.
- Verification: an extension built against a different PyTorch now reports the
  undefined symbol and the rebuild command.
- Commit: 4884d08

## 2026-08-20 — tests/test_wave_backward.py, examples/

- Change: added a regression test for the backward rasterization kernels, a
  dataset-free example (`examples/minimal_render.py`), and a COLMAP walkthrough
  (`examples/train_colmap.md`). Added a gfx1151 section to `README.md`.
- Reason: the wave-size defects in this port share a signature — a warp
  reduction whose width disagrees with the lane arithmetic wrapped around it,
  discarding gradients silently or faulting. Convergence tests do not detect
  the silent case, and `kernel_probe.py` requires an NVIDIA reference machine.
  The new test validates the backward kernels by central finite differences
  against the forward pass, which has no warp-collective reductions and no
  block-size specialisation, so it needs no reference implementation and no
  `nerfacc`. The README described upstream's MI300X target rather than this
  fork's.
- Verification: 11 of 11 pass on gfx1151. Reintroducing a 64-wide warp tile
  against the 32-lane reductions fails 10 of them; the one that still passes is
  the forward tile-size invariance test, which the defect does not affect.
- Commit: 4884d08

## Open items

- The port hardcodes wave32. Detecting the wavefront size at build time and
  injecting it as a macro would let one source tree serve wave32 and wave64
  parts, and would make `GSPLAT_WAVE_SIZE` accurate on a CDNA host rather than
  merely unused.
- The 2DGS and world-space backward kernels carry the same wave32 changes as
  the 3DGS kernel but have no finite-difference coverage.
- `rasterize_bs64_to_pixels_3dgs_bwd_kernel` is disabled on wave32 rather than
  ported. Its 64-lane index arithmetic is unchanged.
- The 2026-08-04 A4000 parity figure of 23.67 dB was measured before the bs64
  path was known to fault at the default tile size. Which kernel that run
  exercised is not established.
