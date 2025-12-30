---
title: "0x12 TMA Descriptor Fence 详解：Device-Side Descriptor 修改与同步"
date: 2024-12-28 10:00:00
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
  - TMA
  - Hopper
  - SM90
  - Descriptor
  - Fence
---

本文详细解析 CUTLASS 中 TMA Descriptor 的 device-side 修改机制，包括三个关键的 fence 函数：`tma_descriptor_cp_fence_release`、`tma_descriptor_fence_release` 和 `tma_descriptor_fence_acquire`。

<!-- more -->

## 背景：为什么需要 Device-Side Descriptor 修改？

在标准 TMA 使用场景中，TMA Descriptor 在 host 端创建（通过 `cuTensorMapEncode`），然后传递给 kernel 使用。但在某些场景下，需要在 device 端动态修改 descriptor：

### Array GEMM (Batched GEMM)
```cpp
// 多个 batch 共享相同的 shape，但地址不同
ptr_A[] = [ptr_A0, ptr_A1, ptr_A2, ...]
ptr_B[] = [ptr_B0, ptr_B1, ptr_B2, ...]
```

每个 batch 处理完后，需要切换到下一个 batch 的地址。

### Grouped GEMM
```cpp
// 每个 group 的矩阵形状可能不同
Group 0: M=1024, N=512, K=256
Group 1: M=2048, N=1024, K=512
Group 2: M=512, N=256, K=128
```

每个 group 需要修改 descriptor 的地址、dims 和 strides。

## 三个 Fence 函数

位于 [include/cute/arch/copy_sm90_desc.hpp](include/cute/arch/copy_sm90_desc.hpp)：

### 1. tma_descriptor_cp_fence_release

```cpp
void tma_descriptor_cp_fence_release(TmaDescriptor const* gmem_desc_ptr,
                                      TmaDescriptor& smem_desc)
{
  asm volatile (
    "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
    :: "l"(gmem_int_desc), "r"(smem_int_desc));
}
```

**关键理解**：这是一个 **SMEM → GMEM** 的拷贝操作！

- `[%0]` = GMEM 地址 (dst)
- `[%1]` = SMEM 地址 (src)
- 指令格式：`tensormap.cp_fenceproxy.global.shared::cta` = 从 `shared::cta` 拷贝到 `global`

**作用**：
1. 将 SMEM 中修改后的 descriptor 拷贝回 GMEM
2. 执行 release fence，确保修改对 TMA Unit 可见

### 2. tma_descriptor_fence_release

```cpp
void tma_descriptor_fence_release()
{
  asm volatile ("fence.proxy.tensormap::generic.release.gpu;");
}
```

**作用**：直接在 GMEM 中修改 descriptor 后使用的 release fence。

### 3. tma_descriptor_fence_acquire

```cpp
void tma_descriptor_fence_acquire(TmaDescriptor const* desc_ptr)
{
  asm volatile (
    "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
    :: "l"(gmem_int_desc)
    : "memory");
}
```

**作用**：Invalidate TMA Unit 的 descriptor cache，确保 TMA Unit 重新读取 GMEM 中的最新 descriptor。

## TMA Unit 的 Descriptor Cache

TMA Unit 是独立于 SM 的硬件单元，有自己专用的 descriptor cache：

```
┌─────────────────────────────────────────────────────────────┐
│                          GPU                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │   SM0   │  │   SM1   │  │   SM2   │  ...                 │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │                      │
│  │ │ L1  │ │  │ │ L1  │ │  │ │ L1  │ │                      │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │                      │
│  └─────────┘  └─────────┘  └─────────┘                      │
│        │            │            │                           │
│        └────────────┼────────────┘                           │
│                     ▼                                        │
│              ┌───────────┐                                   │
│              │    L2     │                                   │
│              └───────────┘                                   │
│                     │                                        │
│        ┌────────────┼────────────┐                           │
│        ▼            ▼            ▼                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │ TMA Unit │ │ TMA Unit │ │ TMA Unit │  (per GPC/TPC)      │
│  │┌────────┐│ │┌────────┐│ │┌────────┐│                     │
│  ││  Desc  ││ ││  Desc  ││ ││  Desc  ││  ← TMA专用缓存      │
│  ││ Cache  ││ ││ Cache  ││ ││ Cache  ││                     │
│  │└────────┘│ │└────────┘│ │└────────┘│                     │
│  └──────────┘ └──────────┘ └──────────┘                     │
│                     │                                        │
│                     ▼                                        │
│              ┌───────────┐                                   │
│              │   GMEM    │  (HBM)                           │
│              └───────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

**关键点**：
- TMA Unit 的 cache 与 SM 的 L1/L2 是独立的
- 普通的 `__threadfence()` 无法影响 TMA Unit 的 cache
- 必须使用 `fence.proxy.tensormap` 来 invalidate TMA 的 descriptor cache

## 完整的 Descriptor 修改流程

### Array GEMM 场景（只修改地址）

```cpp
// 1. 在 SMEM 中修改 descriptor 的地址
tensormap.replace.tile.global_address [smem_desc], new_ptr;

// 2. syncwarp 确保 warp 内所有线程完成修改
__syncwarp();

// 3. cp_fence_release: SMEM→GMEM + release fence
tensormap.cp_fenceproxy.global.shared::cta [gmem_desc], [smem_desc], 128;

// 4. fence_acquire: invalidate TMA Unit 的 descriptor cache
fence.proxy.tensormap::generic.acquire.gpu [gmem_desc], 128;

// 5. 现在可以用更新后的 descriptor 执行 TMA 操作
cp.async.bulk.tensor ...
```

### Grouped GEMM 场景（修改地址 + dims + strides）

```cpp
// 1. 修改地址
tensormap.replace.tile.global_address [smem_desc], new_ptr;

// 2. 修改维度
tensormap.replace.tile.global_dim [smem_desc], dim0, new_dim0;
tensormap.replace.tile.global_dim [smem_desc], dim1, new_dim1;
...

// 3. 修改步长
tensormap.replace.tile.global_stride [smem_desc], ord0, new_stride0;
tensormap.replace.tile.global_stride [smem_desc], ord1, new_stride1;
...

// 4-6. 同上的同步流程
```

## CUTLASS 中的实现

### tensormaps_replace_global_address

只修改地址，用于 Array GEMM：

```cpp
// sm90_mma_array_tma_gmma_ss_warpspecialized.hpp:660-665
void tensormaps_replace_global_address(..., int32_t next_batch) {
  cute::tma_descriptor_replace_addr_in_shared_mem(
      shared_tensormaps.smem_tensormap_A,
      mainloop_params.ptr_A[next_batch]);
  cute::tma_descriptor_replace_addr_in_shared_mem(
      shared_tensormaps.smem_tensormap_B,
      mainloop_params.ptr_B[next_batch]);
}
```

### tensormaps_replace_global_tensor_properties

修改 dims 和 strides，用于 Grouped GEMM：

```cpp
// sm90_mma_array_tma_gmma_ss_warpspecialized.hpp:671-710
void tensormaps_replace_global_tensor_properties(...) {
  const uint32_t M = get<0>(problem_shape_mnkl);
  const uint32_t N = get<1>(problem_shape_mnkl);
  const uint32_t K = get<2>(problem_shape_mnkl);

  // 构建新的 shape 和 stride
  cute::array<uint32_t, 5> prob_shape_A  = {1,1,1,1,1};
  cute::array<uint64_t, 5> prob_stride_A = {0,0,0,0,0};

  // 从参数获取该 group 的 stride
  Tensor tensor_a = make_tensor(ptr_A, make_shape(M,K,Int<1>{}),
                                mainloop_params.dA[next_group]);

  // 填充 shape 和 stride
  cute::detail::fill_tma_gmem_shape_stride(*observed_tma_load_a_, tensor_a,
      prob_shape_A, prob_stride_A);

  // 用 tensormap.replace 指令修改 descriptor
  cute::tma_descriptor_replace_dims_strides_in_shared_mem(
      shared_tensormaps.smem_tensormap_A,
      prob_shape_A,
      prob_stride_A);
}
```

### 完整的 tensormaps_perform_update

```cpp
void tensormaps_perform_update(..., int32_t next_batch) {
  if (cute::elect_one_sync()) {
    // 1. 修改地址
    tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

    // 2. 如果是 Grouped GEMM，还需要修改 dims 和 strides
    if constexpr (IsGroupedGemmKernel) {
      tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_mnkl);
    }
  }
}
```

### tensormaps_cp_fence_release

```cpp
void tensormaps_cp_fence_release(TensorMapStorage& shared_tensormaps,
    cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
  if (cute::elect_one_sync()) {
    cute::tma_desc_commit_group();
    cute::tma_desc_wait_group();
  }
  // 整个 warp 必须执行这个操作（对齐要求）
  tma_descriptor_cp_fence_release(get<0>(input_tensormaps),
                                   shared_tensormaps.smem_tensormap_A);
  tma_descriptor_cp_fence_release(get<1>(input_tensormaps),
                                   shared_tensormaps.smem_tensormap_B);
}
```

### tensormaps_fence_acquire

```cpp
void tensormaps_fence_acquire(cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
  cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
  cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
}
```

## 数据流总结

```
Host GMEM descriptor (初始)
        │
        ▼ (初始拷贝到SMEM，kernel启动时)
SMEM descriptor
        │
        ▼ tensormap.replace (在SMEM中修改)
SMEM descriptor (修改后)
        │
        ▼ tensormap.cp_fenceproxy (SMEM→GMEM + release fence)
GMEM descriptor (更新后)
        │
        ▼ fence.proxy.tensormap.acquire (invalidate TMA cache)
TMA Unit 使用更新后的 descriptor
```

## 两种场景对比

| 场景 | 每个 batch/group 的矩阵 | 需要修改的字段 | 函数调用 |
|------|-------------------------|----------------|----------|
| **Array GEMM** | 形状相同，地址不同 | `global_address` | `tensormaps_replace_global_address` |
| **Grouped GEMM** | 形状可能不同，地址不同 | `global_address` + `dims` + `strides` | `tensormaps_replace_global_address` + `tensormaps_replace_global_tensor_properties` |

## 为什么要这样设计？

**节省 host 端创建 descriptor 的开销**：
- TMA descriptor 创建是 host 操作，需要调用 `cuTensorMapEncode`
- 如果有 1000 个 batch/group，需要创建 1000 个 descriptor，开销很大
- 通过在 device 端动态修改地址/dims/strides，只需要创建 **1 个** descriptor 模板

## PTX 指令参考

### tensormap.replace

```ptx
// 修改地址
tensormap.replace.tile.global_address.shared::cta.b1024.b64 [smem_desc], new_addr;

// 修改维度
tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [smem_desc], dim_idx, new_dim;

// 修改步长
tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [smem_desc], ord_idx, new_stride;
```

### tensormap.cp_fenceproxy

```ptx
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [gmem], [smem], 128;
```

### fence.proxy.tensormap

```ptx
// Release fence
fence.proxy.tensormap::generic.release.gpu;

// Acquire fence
fence.proxy.tensormap::generic.acquire.gpu [gmem_desc], 128;
```

## References

- [CUTLASS Source: copy_sm90_desc.hpp](include/cute/arch/copy_sm90_desc.hpp)
- [CUTLASS Source: sm90_mma_array_tma_gmma_ss_warpspecialized.hpp](include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp)
- [PTX ISA: tensormap.cp_fenceproxy](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [libcudacxx: tensormap.cp_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap.cp_fenceproxy.html)
