#pragma once

#include <algorithm>
#include <cstdint>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>
#include <glm/gtc/type_ptr.hpp>

// Wavefront size of the target GPU (wave32 on RDNA gfx10xx/gfx11xx, wave64 on
// CDNA/GCN gfx9xx). Normally injected by setup.py from rocminfo; this fallback
// keeps manual compilation working.
#if defined(USE_ROCM) && !defined(GSPLAT_WARP_SIZE)
#define GSPLAT_WARP_SIZE 64
#endif

#ifndef USE_ROCM
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <c10/hip/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

#else
#include <hip/hip_cooperative_groups.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/Atomic.cuh>
#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_reduce.hpp>
#include <rocprim/warp/warp_reduce.hpp>
#endif

namespace gsplat {

#ifndef USE_ROCM
// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)
#else
#define cub hipcub
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        auto res = func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto temp_storage = at::empty( \
            {static_cast<int64_t>(temp_storage_bytes)}, \
            at::TensorOptions().dtype(at::kByte).device(at::kCUDA)); \
        res = func(temp_storage.data_ptr(), temp_storage_bytes, __VA_ARGS__);  \
	TORCH_CHECK(res == hipSuccess, "rocPRIM call failed: ",              \
            hipGetErrorString(res));                                          \
    } while (false)
#endif
} // namespace gsplat
