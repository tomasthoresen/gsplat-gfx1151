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
