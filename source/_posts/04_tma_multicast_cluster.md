---
title: TMA Multicast 与 Cluster 数据共享
date: 2024-12-24
categories: [CUDA, CUTLASS, TMA]
tags: [multicast, cluster, tma, gemm]
---

# TMA Multicast 与 Cluster 数据共享

## 1. 引言

在 Hopper 架构的 GEMM 中，TMA Multicast 可以将一份数据同时写入多个 CTA 的 SMEM，大幅节省全局内存带宽。本文详细解析 Cluster 中的数据共享机制。

## 2. GEMM 中的数据需求

### 2.1 基本分工

```
C[M, N] = A[M, K] × B[K, N]

2×2 Cluster 的 CTA 分工:
              N 方向
         N_tile_0    N_tile_1
        ┌───────────┬───────────┐
M_tile_0│ CTA(0,0)  │ CTA(0,1)  │  ← 这两个需要相同的 A[M_tile_0]
        ├───────────┼───────────┤
M_tile_1│ CTA(1,0)  │ CTA(1,1)  │  ← 这两个需要相同的 A[M_tile_1]
        └───────────┴───────────┘
             ↑           ↑
          需要相同    需要相同
          B[N_tile_0] B[N_tile_1]
```

### 2.2 数据共享关系

| 矩阵 | 共享规则 | 示例 |
|------|----------|------|
| A | 同一行的 CTA 共享 | CTA(0,0), CTA(0,1) 共享 A[M_tile_0] |
| B | 同一列的 CTA 共享 | CTA(0,0), CTA(1,0) 共享 B[N_tile_0] |

## 3. Multicast 加载分工

### 3.1 加载策略

```
A 矩阵 (第一列 CTA 负责加载):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  CTA(0,0) 加载 A[M_tile_0, K]                               │
│      │                                                      │
│      └──→ multicast ──→ CTA(0,0), CTA(0,1) 的 SMEM         │
│                                                             │
│  CTA(1,0) 加载 A[M_tile_1, K]                               │
│      │                                                      │
│      └──→ multicast ──→ CTA(1,0), CTA(1,1) 的 SMEM         │
│                                                             │
│  CTA(0,1), CTA(1,1): 不加载 A (从 multicast 接收)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

B 矩阵 (第一行 CTA 负责加载):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  CTA(0,0) 加载 B[K, N_tile_0]                               │
│      │                                                      │
│      └──→ multicast ──→ CTA(0,0), CTA(1,0) 的 SMEM         │
│                                                             │
│  CTA(0,1) 加载 B[K, N_tile_1]                               │
│      │                                                      │
│      └──→ multicast ──→ CTA(0,1), CTA(1,1) 的 SMEM         │
│                                                             │
│  CTA(1,0), CTA(1,1): 不加载 B (从 multicast 接收)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 每个 CTA 的工作量

| CTA | 加载 A | 加载 B | 说明 |
|-----|--------|--------|------|
| CTA(0,0) | ✓ | ✓ | 最忙，加载两份数据 |
| CTA(0,1) | ✗ | ✓ | 只加载 B |
| CTA(1,0) | ✓ | ✗ | 只加载 A |
| CTA(1,1) | ✗ | ✗ | 不加载，全部 multicast |

**带宽节省：A 加载 2 次 (vs 4 次)，B 加载 2 次 (vs 4 次) → 节省 50%！**

## 4. 数据流向图示

```
                    Global Memory
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    A[M_tile_0]    A[M_tile_1]    B[N_tile_0]    B[N_tile_1]
          │              │              │              │
          │              │              │              │
     CTA(0,0)       CTA(1,0)       CTA(0,0)       CTA(0,1)
     加载 A0        加载 A1        加载 B0        加载 B1
          │              │              │              │
          │              │              │              │
    ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
    │           │  │           │  │           │  │           │
    ▼           ▼  ▼           ▼  ▼           ▼  ▼           ▼
 CTA(0,0)   CTA(0,1) CTA(1,0) CTA(1,1) CTA(0,0) CTA(1,0) CTA(0,1) CTA(1,1)
  SMEM_A     SMEM_A  SMEM_A   SMEM_A   SMEM_B   SMEM_B   SMEM_B   SMEM_B
```

## 5. Multicast Mask 计算

### 5.1 CTA 编号

```
2×2 Cluster CTA 编号 (m + n * ClusterM):
        ┌───────┬───────┐
        │  0    │   2   │   (注意: 按列主序)
        ├───────┼───────┤
        │  1    │   3   │
        └───────┴───────┘
```

### 5.2 Mask 值

```cpp
// A 的 multicast mask (横向, 同一行):
// Row 0: CTA 0, 2 → mask = 0b0101 = 5
// Row 1: CTA 1, 3 → mask = 0b1010 = 10

// B 的 multicast mask (纵向, 同一列):
// Col 0: CTA 0, 1 → mask = 0b0011 = 3
// Col 1: CTA 2, 3 → mask = 0b1100 = 12
```

### 5.3 代码实现

```cpp
// 判断当前 CTA 是否需要加载
bool should_load_A = (cluster_coord_n == 0);  // 第一列 CTA
bool should_load_B = (cluster_coord_m == 0);  // 第一行 CTA

// 计算 multicast mask
if (should_load_A) {
    // A: multicast 给同一行的所有 CTA
    for (int n = 0; n < ClusterN; ++n) {
        mcast_mask_A |= (1 << (cluster_coord_m + n * ClusterM));
    }
}

if (should_load_B) {
    // B: multicast 给同一列的所有 CTA
    for (int m = 0; m < ClusterM; ++m) {
        mcast_mask_B |= (1 << (m + cluster_coord_n * ClusterM));
    }
}
```

## 6. TMA Multicast 加载代码

### 6.1 Producer Load 逻辑

```cpp
CUTLASS_DEVICE void producer_load() {
    // 1. 获取 cluster 内坐标
    auto [cta_m, cta_n] = cluster_local_block_id();
    
    // 2. 判断是否负责加载
    bool load_A = (cta_n == 0);  // 第一列加载 A
    bool load_B = (cta_m == 0);  // 第一行加载 B
    
    // 3. 计算 multicast mask
    uint16_t mask_A = load_A ? row_mask(cta_m) : 0;
    uint16_t mask_B = load_B ? col_mask(cta_n) : 0;
    
    // 4. 发起 TMA load
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        pipeline.producer_acquire(state);
        
        if (load_A) {
            tma_load_multicast(smem_A, desc_A, mask_A, barrier);
        }
        if (load_B) {
            tma_load_multicast(smem_B, desc_B, mask_B, barrier);
        }
        
        pipeline.producer_commit(state);
    }
}
```

### 6.2 TMA Multicast PTX

```cpp
CUTE_DEVICE void
tma_load_multicast(void* smem_ptr, 
                   TmaDescriptor const& desc,
                   uint16_t mcast_mask,
                   uint64_t* mbar_ptr)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :
        : "r"(smem_ptr), "l"(desc), "r"(mbar_ptr), 
          "r"(coord_x), "r"(coord_y), "h"(mcast_mask)
        : "memory"
    );
}
```

## 7. Consumer Release 的 Multicast Arrive

### 7.1 问题

Producer 发起的 TMA multicast 会写入多个 CTA 的 SMEM，必须等所有这些 CTA 的 Consumer 消费完才能重用 buffer。

### 7.2 解决方案

每个 Consumer warpgroup 向所有有效 CTA arrive：

```cpp
// 构造函数中设置
auto [is_signaling_thread, dst_blockid] = 
    spread_arrivals_to_warpgroup(thread_idx % NumThreadsPerWarpGroup, warp_idx);

is_signaling_thread_ &= is_same_row_or_col(dst_blockid_, block_id, cluster_shape);

// 运行时 arrive
void consumer_release(stage) {
    empty_barrier.arrive(dst_blockid_, is_signaling_thread_);
}
```

### 7.3 spread_arrivals_to_warpgroup

```cpp
cute::tuple<bool, uint32_t> spread_arrivals_to_warpgroup(
    int thread_idx_in_warpgroup, int warp_idx)
{
    // 每 8 个线程选一个 signaling thread
    // 128 threads / 16 CTA = 8
    bool is_signaling_thread = (thread_idx_in_warpgroup % 8) == 0;
    
    // 用 Swizzle Layout 分配 dst_blockid (0-15)
    auto layout = cute::composition(Swizzle<2,0,-2>{},
                                    Layout<Shape<_4,_4>,Stride<_4,_1>>{});
    uint32_t thread_row = warp_idx % 4;
    uint32_t thread_col = (thread_idx_in_warpgroup / 8) % 4;
    uint32_t dst_blockid = layout(thread_row, thread_col);
    
    return cute::make_tuple(is_signaling_thread, dst_blockid);
}
```

### 7.4 is_same_row_or_col 过滤

```cpp
bool is_same_row_or_col(int dst_block_id, dim3 block_id, ClusterShape cluster_shape) {
    int dst_m = dst_block_id % size<0>(cluster_shape);
    int dst_n = dst_block_id / size<0>(cluster_shape);
    return (dst_m == block_id.x) || (dst_n == block_id.y);
}
```

## 8. 2×2 Cluster Arrive 流程

```
CTA(0,0) 的 Consumer Warpgroup 执行 consumer_release():

                    Warpgroup 中 16 个 signaling threads
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         CTA(0,0)        CTA(0,1)        CTA(1,0)         CTA(1,1)
         barrier         barrier         barrier          barrier
            ↑               ↑               ↑                ✗
            │               │               │           (不是同行/同列)
         arrive          arrive          arrive
         
    is_same_row_or_col:  ✓ (self)    ✓ (same row)   ✓ (same col)    ✗

每个 warpgroup 向 3 个有效 CTA 各发送 1 次 arrive
2 个 warpgroups → 每个有效 CTA 收到 2 次 arrive
3 个有效 CTA × 2 warpgroups = 6 total arrives
```

## 9. 远程 Barrier Arrive PTX

```cpp
static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    if (pred) {
        asm volatile(
            "{\n\t"
            ".reg .b32 remAddr32;\n\t"
            "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"  // 映射远程地址
            "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
            "}"
            :
            : "r"(smem_addr), "r"(cta_id));
    }
}
```

**关键指令：**
- `mapa.shared::cluster`: 本地 SMEM 地址 → 远程 CTA SMEM 地址
- `mbarrier.arrive.shared::cluster`: 对远程 barrier 执行 arrive

## 10. 总结

### 10.1 Multicast 规则

| 矩阵 | 加载者 | Multicast 方向 | Mask |
|------|--------|----------------|------|
| A | 第一列 CTA | 横向 (同一行) | row_mask |
| B | 第一行 CTA | 纵向 (同一列) | col_mask |

### 10.2 Consumer Arrival Count

```cpp
// 非 multicast
arrival_count = num_consumers;  // per-thread arrive

// Multicast
arrival_count = (ClusterM + ClusterN - 1) * num_warpgroups;  // per-warpgroup
```

### 10.3 核心优势

1. **带宽节省**：每份数据只从 GMEM 读一次
2. **负载均衡**：不同 CTA 负责不同数据
3. **正确同步**：通过 remote barrier arrive 确保所有接收者消费完成
