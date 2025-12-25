---
title: "0x10 CUTLASS SM90 WarpGroup MMA API 详解"
date: 2024-12-24
categories:
  - CUTLASS
tags:
  - CUTLASS
  - WGMMA
  - SM90
  - Hopper
  - Pipeline
---

本文详解 CUTLASS SM90 WarpGroup MMA 的核心 API，基于 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 源码分析 WGMMA 异步执行模型和同步原语的使用方法。

<!-- more -->

> **示例代码**: [0x10_warpgroup_mma_api.cu](https://github.com/DrXuQian/cute-examples/blob/main/0x10_warpgroup_mma_api.cu)

> **核心 API 速览**
> 1. **warpgroup_arrive()**: 发起 WGMMA fence，标记操作开始
> 2. **warpgroup_commit_batch()**: 提交一批 WGMMA 操作
> 3. **warpgroup_wait\<N\>()**: 等待直到最多 N 批操作在飞
> 4. **warpgroup_fence_operand()**: 防止累加器被优化重排

## 1. WGMMA 异步执行模型

### 1.1 WarpGroup MMA 概述

Hopper (SM90) 引入了 **WarpGroup MMA (WGMMA)**，允许 4 个 warp（128 线程）协作执行大规模矩阵乘法。关键特性：

- **异步执行**: MMA 操作在后台异步执行
- **批处理**: 多个 MMA 操作可组成一批
- **流水线**: 支持多批操作同时在飞

```
WarpGroup (128 threads = 4 warps)
+-- Warp 0 (32 threads)
+-- Warp 1 (32 threads)
+-- Warp 2 (32 threads)
+-- Warp 3 (32 threads)
    |
    v
  WGMMA Unit (异步执行 MMA)
```

### 1.2 异步执行流程

```cpp
// 1. Fence - 标记 MMA 操作边界
warpgroup_arrive();

// 2. 发射 MMA 指令
cute::gemm(tiled_mma, tCrA, tCrB, accum);

// 3. Commit - 提交一批操作
warpgroup_commit_batch();

// 4. Wait - 等待完成
warpgroup_wait<0>();  // 等待所有操作完成
```

## 2. 核心同步原语

### 2.1 warpgroup_arrive()

```cpp
// 源码: cute/arch/mma_sm90_gmma.hpp:47-57
CUTE_HOST_DEVICE void warpgroup_arrive() {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
```

**作用**:
- 发射 `wgmma.fence.sync.aligned` PTX 指令
- 标记 WGMMA 操作的开始边界
- 确保之前的内存操作完成

**使用时机**: 在发射 `gemm()` 之前调用

### 2.2 warpgroup_commit_batch()

```cpp
// 源码: cute/arch/mma_sm90_gmma.hpp:74-84
CUTE_HOST_DEVICE void warpgroup_commit_batch() {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
```

**作用**:
- 发射 `wgmma.commit_group.sync.aligned` PTX 指令
- 将当前未提交的 WGMMA 操作打包成一批
- 批次编号自动递增

**使用时机**: 在 `gemm()` 之后、下一轮 `warpgroup_arrive()` 之前调用

### 2.3 warpgroup_wait\<N\>()

```cpp
// 源码: cute/arch/mma_sm90_gmma.hpp:59-71
template <int N>
CUTE_HOST_DEVICE void warpgroup_wait() {
  static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
#endif
}
```

**作用**:
- 发射 `wgmma.wait_group.sync.aligned N` PTX 指令
- 阻塞直到**最多 N 批**操作还在执行
- N=0 表示等待所有操作完成

**参数范围**: N ∈ [0, 7]

| N 值 | 含义 |
|------|------|
| 0 | 等待所有批次完成 |
| 1 | 允许最多 1 批在飞 |
| 2 | 允许最多 2 批在飞 |
| ... | ... |
| 7 | 允许最多 7 批在飞 |

### 2.4 warpgroup_fence_operand()

```cpp
// 源码: cute/arch/mma_sm90_gmma.hpp:86-103
CUTE_HOST_DEVICE void warpgroup_fence_operand(float& reg) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+f"(reg) :: "memory");
#endif
}

CUTE_HOST_DEVICE void warpgroup_fence_operand(uint32_t& reg) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(reg) :: "memory");
#endif
}
```

**作用**:
- 编译器屏障，防止累加器寄存器被优化重排
- 使用 `"+f"` (float) 或 `"+r"` (uint32) 约束
- 确保 WGMMA 操作看到正确的累加器值

**使用时机**: 在 `warpgroup_arrive()` 之前和 `warpgroup_wait()` 之后

## 3. WGMMA 批处理模型

### 3.1 批处理流程图

```
时间 ->

Thread:  fence -> gemm -> gemm -> commit -> fence -> gemm -> commit -> wait<1> -> wait<0>
                  |______Batch 0______|              |__Batch 1__|
                                                                    ^
                                                                    |
WGMMA:                [===Batch 0===]                [===Batch 1===]
                      执行中                          执行中
                                       ^                            ^
                                       |                            |
                                 commit后开始执行            wait<0>时全部完成
```

### 3.2 K_PIPE_MMAS 控制

```cpp
// 源码: sm90_mma_tma_gmma_ss_warpspecialized.hpp:264
static constexpr int K_PIPE_MMAS = 1;

// 使用: 在主循环中控制在飞批次数
warpgroup_wait<K_PIPE_MMAS>();  // 允许 1 批在飞
```

K_PIPE_MMAS 控制了 MMA 操作的流水线深度：
- 较大值: 更多并行，更高吞吐
- 较小值: 更低延迟，更少资源占用

## 4. CollectiveMma::mma() 完整流程

### 4.1 代码结构

```cpp
// 源码: sm90_mma_tma_gmma_ss_warpspecialized.hpp:416-559
template <class FrgTensorC>
CUTLASS_DEVICE void mma(
    MainloopPipeline pipeline,
    PipelineState smem_pipe_read,
    FrgTensorC& accum,
    int k_tile_count,
    int thread_idx,
    TensorStorage& shared_tensors,
    Params const& mainloop_params)
{
  // Step 1: 创建 SMEM tensors
  Tensor sA = make_tensor(make_smem_ptr(...), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(...), SmemLayoutB{});

  // Step 2: 获取当前 warp group 的 MMA slice
  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_slice(warp_group_idx);

  // Step 3: Partition A/B for this thread
  Tensor tCsA = thread_mma.partition_A(sA);
  Tensor tCsB = thread_mma.partition_B(sB);

  // Step 4: 创建 fragment/descriptor
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);

  // Step 5: Prologue + Mainloop
  // ...
}
```

### 4.2 Prologue (第一个 K-tile)

```cpp
// 源码: sm90_mma_tma_gmma_ss_warpspecialized.hpp:479-503
tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;  // 清零累加器
warpgroup_fence_operand(accum);

{
  // 等待 SMEM 数据就绪
  auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
  pipeline.consumer_wait(smem_pipe_read, barrier_token);

  int read_stage = smem_pipe_read.index();

  warpgroup_arrive();  // <- Fence
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

  // K-内层循环
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
    cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage),
                          tCrB(_,_,k_block,read_stage), accum);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;  // 后续累加
  }

  warpgroup_commit_batch();  // <- Commit
  ++smem_pipe_read;
}
```

**关键点**:
- `ScaleOut::Zero`: 第一次 MMA 清零 D
- `ScaleOut::One`: 后续 MMA 累加到 D
- 内层 K 循环遍历 SMEM 中的 K blocks

### 4.3 Mainloop (剩余 K-tiles)

```cpp
// 源码: sm90_mma_tma_gmma_ss_warpspecialized.hpp:528-556
CUTLASS_PRAGMA_NO_UNROLL
for ( ; k_tile_count > 0; --k_tile_count) {
  // 等待 SMEM 数据
  auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
  pipeline.consumer_wait(smem_pipe_read, barrier_token);

  int read_stage = smem_pipe_read.index();

  warpgroup_fence_operand(accum);
  warpgroup_arrive();

  // 执行 MMA
  cute::gemm(tiled_mma, tCrA(_,_,_,read_stage),
                        tCrB(_,_,_,read_stage), accum);

  warpgroup_commit_batch();

  // 等待前面的批次完成，释放 SMEM
  warpgroup_wait<K_PIPE_MMAS>();
  warpgroup_fence_operand(accum);

  // 释放已消费的 SMEM stage
  pipeline.consumer_release(smem_pipe_release);

  ++smem_pipe_read;
  ++smem_pipe_release;
}
```

### 4.4 流程图

```
                    +-- consumer_wait (等待 TMA 完成) --+
                    |                                   |
                    v                                   |
+-- fence_operand --+-- arrive --+-- gemm --+-- commit --+-- wait<1> --+-- fence_operand --+
|                                                                      |                    |
|                    <- 批次 N-1 可能在执行 ->        <- 批次 N 提交 ->  |                    |
|                                                                      v                    |
+-- consumer_release (释放 SMEM stage N-K_PIPE_MMAS) <-----------------+                    |
                                                                                            |
                    +-----------------------------------------------------------------------+
                    |
                    v
              下一轮迭代
```

## 5. GMMA ScaleOut 模式

### 5.1 ScaleOut 枚举

```cpp
// 源码: cute/arch/mma_sm90_gmma.hpp:112-115
enum class ScaleOut {
  Zero = 0,  // D = A * B        (清零累加)
  One  = 1   // D = A * B + D    (累加模式)
};
```

### 5.2 使用场景

| 模式 | 公式 | 使用场景 |
|------|------|----------|
| Zero | D = A × B | K 循环第一次迭代，初始化累加器 |
| One | D = A × B + D | K 循环后续迭代，累加结果 |

```cpp
// Prologue: 第一个 MMA
tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
cute::gemm(tiled_mma, ...);  // D = A * B

// 后续: 累加
tiled_mma.accumulate_ = GMMA::ScaleOut::One;
cute::gemm(tiled_mma, ...);  // D = A * B + D
```

## 6. SS vs RS 模式

### 6.1 SS 模式 (Shared-Shared)

```cpp
// A 和 B 都从 SMEM 读取，使用 descriptor
struct MMA_64x8x16_F16F16F16_SS {
  using ARegisters = uint64_t[1];  // SMEM descriptor
  using BRegisters = uint64_t[1];  // SMEM descriptor
  using CRegisters = uint32_t[2];  // Register

  static void fma(uint64_t const& desc_a,
                  uint64_t const& desc_b,
                  uint32_t& d0, uint32_t& d1,
                  GMMA::ScaleOut scale_D);
};
```

### 6.2 RS 模式 (Register-Shared)

```cpp
// A 从 Register 读取，B 从 SMEM 读取
struct MMA_64x8x16_F16F16F16_RS {
  using ARegisters = uint32_t[4];  // 4 个 32-bit 寄存器
  using BRegisters = uint64_t[1];  // SMEM descriptor
  using CRegisters = uint32_t[2];  // Register

  static void fma(uint32_t const& a0, uint32_t const& a1,
                  uint32_t const& a2, uint32_t const& a3,
                  uint64_t const& desc_b,
                  uint32_t& d0, uint32_t& d1,
                  GMMA::ScaleOut scale_D);
};
```

### 6.3 模式选择

| 特性 | SS 模式 | RS 模式 |
|------|---------|---------|
| A 来源 | SMEM (descriptor) | Register |
| B 来源 | SMEM (descriptor) | SMEM (descriptor) |
| 适用场景 | TMA 直接加载到 SMEM | 需要 A 变换/预处理 |
| 本文件 | ✓ (主要使用) | - |

## 7. Pipeline 集成

### 7.1 Pipeline 状态机

```cpp
// Consumer 视角的 Pipeline 操作
pipeline.consumer_try_wait(smem_pipe_read);  // 尝试获取 barrier token
pipeline.consumer_wait(smem_pipe_read, token);  // 等待数据就绪
// ... 执行 MMA ...
pipeline.consumer_release(smem_pipe_release);  // 释放 stage
```

### 7.2 Pipeline 与 WGMMA 同步

```
Producer (TMA Load)                    Consumer (WGMMA)
       |                                     |
       v                                     v
  producer_acquire                    consumer_try_wait
       |                                     |
  TMA copy to SMEM                    consumer_wait
       |                                     |
  [TMA complete_tx]  -----barrier----> [ready]
       |                                     |
  ++smem_pipe_write                   warpgroup_arrive
                                             |
                                        gemm + commit
                                             |
                                      warpgroup_wait
                                             |
                                      consumer_release
                                             |
                           <---barrier---- [释放]
                                             |
                                      ++smem_pipe_read
```

## 8. 完整代码示例

```cpp
// 简化版 WGMMA 主循环
template <class FrgTensorC>
CUTLASS_DEVICE void mma_simplified(
    Pipeline pipeline,
    PipelineState smem_pipe_read,
    FrgTensorC& accum,
    int k_tile_count,
    TensorStorage& shared)
{
  Tensor sA = make_tensor(..., SmemLayoutA{});
  Tensor sB = make_tensor(..., SmemLayoutB{});

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_slice(warp_group_idx);

  Tensor tCrA = thread_mma.make_fragment_A(thread_mma.partition_A(sA));
  Tensor tCrB = thread_mma.make_fragment_B(thread_mma.partition_B(sB));

  PipelineState smem_pipe_release = smem_pipe_read;

  // Prologue: 第一个 k-tile
  {
    pipeline.consumer_wait(smem_pipe_read,
                           pipeline.consumer_try_wait(smem_pipe_read));
    int stage = smem_pipe_read.index();

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    cute::gemm(tiled_mma, tCrA(_,_,_,stage), tCrB(_,_,_,stage), accum);
    warpgroup_commit_batch();

    ++smem_pipe_read;
  }

  tiled_mma.accumulate_ = GMMA::ScaleOut::One;

  // Mainloop: 剩余 k-tiles
  for (int k = k_tile_count - 1; k > 0; --k) {
    pipeline.consumer_wait(smem_pipe_read,
                           pipeline.consumer_try_wait(smem_pipe_read));
    int stage = smem_pipe_read.index();

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA(_,_,_,stage), tCrB(_,_,_,stage), accum);
    warpgroup_commit_batch();

    warpgroup_wait<1>();  // 允许 1 批在飞
    warpgroup_fence_operand(accum);

    pipeline.consumer_release(smem_pipe_release);
    ++smem_pipe_read;
    ++smem_pipe_release;
  }

  // Epilogue: 等待所有完成
  warpgroup_wait<0>();

  // 释放剩余 stages
  pipeline.consumer_release(smem_pipe_release);
}
```

## 9. 总结

| API | PTX 指令 | 作用 |
|-----|----------|------|
| `warpgroup_arrive()` | `wgmma.fence.sync.aligned` | 标记 MMA 批次开始 |
| `warpgroup_commit_batch()` | `wgmma.commit_group.sync.aligned` | 提交当前批次 |
| `warpgroup_wait<N>()` | `wgmma.wait_group.sync.aligned N` | 等待至最多 N 批在飞 |
| `warpgroup_fence_operand()` | 编译器屏障 | 防止累加器重排 |

**调用顺序**:
```
fence_operand -> arrive -> gemm -> commit -> wait -> fence_operand -> release
```

## 参考资料

- 源码: [sm90_mma_tma_gmma_ss_warpspecialized.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp)
- 源码: [cute/arch/mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp)
- [NVIDIA PTX ISA - wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/)
