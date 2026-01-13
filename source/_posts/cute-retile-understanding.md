---
title: CuTe TiledCopy 中的 retile 机制深度解析
date: 2025-01-13 22:07:00
tags:
  - CUDA
  - CUTLASS
  - CuTe
  - GPU
categories:
  - CUTLASS
---

# CuTe TiledCopy 中的 retile 机制深度解析

## 1. 问题背景

在 CUTLASS 3.x 的 epilogue 中，需要将 MMA 计算的结果（存储在寄存器中）写入 shared memory。这个过程涉及 `make_tiled_copy_C` 和 `retile_S` 操作。

```cpp
// sm90_epilogue_tma_warpspecialized.hpp:706-707
Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);  // ((R2S,R2S_V),MMA_M,MMA_N)
Tensor tRS_sD   = thread_r2s.partition_D(sD_epi);     // (R2S,R2S_M,R2S_N,PIPE_D)
```

## 2. 核心问题：MMA 输出与 Copy 指令的 layout 不匹配

### 2.1 TiledMMA 的输出 layout

TiledMMA 的 `get_layoutC_TV()` 返回的 value 部分结构是：

```
(thr_idx, (FrgV, (RestM, RestN))) -> (M, N)
```

其中：
- `FrgV`: 单个 MMA atom 每个线程的 values（如 SM80 的 `(2,2)` = 4 个 values）
- `RestM`: M 方向上 atom 的重复次数
- `RestN`: N 方向上 atom 的重复次数

**具体例子**：假设 TiledMMA 配置为：
- Atom: `SM80_16x8x16`，每线程 4 个 values，布局 `(2,2)`
- ThrLayout: `(2, 2, 1)` - 2x2 的线程 tile
- Tile size: 64x64

那么每个线程的 value shape 为：
```
((2,2), (2, 4)) = 32 个 values
  ↑       ↑
FrgV  (RestM, RestN)
```

### 2.2 Copy 指令的期望格式

不同的 copy 指令期望不同的 value 分组方式：

| Copy Op | PTX 指令 | 每次处理元素数 |
|---------|----------|---------------|
| `SM90_U16x8_STSM_T` | `stmatrix.x4.trans` | 8 个 FP16 |
| `SM90_U16x4_STSM_T` | `stmatrix.x2.trans` | 4 个 FP16 |
| `SM90_U32x4_STSM_N` | `stmatrix.x4` | 8 个 FP16 |
| AutoVectorizing | `st.shared.v2.b32` | 4 个 FP16 |

Copy 指令期望的格式是 `(atom_vals, rest_vals)`：
- `atom_vals`: 一次 copy 指令处理的元素数
- `rest_vals`: 需要执行的 copy 指令次数

### 2.3 不匹配的本质

```
MMA 输出:  ((2,2), (2, 4))  - 嵌套的多维结构
Copy 期望: (8, 4)           - 扁平的二维结构
```

虽然元素总数相同（32个），但组织方式不同，直接使用会导致 `cute::copy` 无法正确迭代。

## 3. retile 的作用

### 3.1 `cute::copy` 的迭代模式

从源码 `copy.hpp:224-227`：

```cpp
CUTE_UNROLL
for (int i = 0; i < size<1>(dst_c); ++i) {
  copy_atom.call(src_c(_,i), dst_c(_,i));  // 每次处理第一个 mode 的所有 values
}
```

`cute::copy` 会：
1. 对第一个 mode（`atom_vals`）调用 `copy_atom.call`
2. 对其余 modes 进行循环迭代

### 3.2 retile 的转换

`retile_S` 把 MMA 的嵌套 value 布局转换成 Copy Atom 期望的格式：

```
retile 前: ((2,2), (2, 4))     // MMA 的嵌套布局
retile 后: (8, 4)              // Copy Atom 期望的格式
           ↑  ↑
    atom_vals  rest_vals
```

### 3.3 retile 的实现原理

从 `copy_atom.hpp:298-302`：

```cpp
// (m,n) -> v_idx -- The shape and order of the V inside of TiledLayout_TV
auto frg_layout_mn = upcast<TiledNumThr{} * V>(
    right_inverse(TiledLayout_TV{}).with_shape(shape(Tiler_MN{})));

// (atom_vals,rest_vals) -> (v,m,n)
auto frg_layout_v = zipped_divide(
    logical_product(make_layout(V), right_inverse(frg_layout_mn)),
    make_layout(AtomNumVal{}));
```

核心步骤：
1. 从 MMA 的 TV layout 中提取 value 的排列顺序
2. 按照 Copy Atom 的 `AtomNumVal` 进行分组
3. 重新组织成 `(atom_vals, rest_vals)` 的形式

## 4. 实际应用案例

### 4.1 stmatrix 指令的 retile

**场景**：FP16 RowMajor 输出，EpilogueTile = (64, 64)

根据 `sm90_common.inl:42-52` 的选择逻辑：
```cpp
if constexpr (sizeof(ElementD) == 2 && size<0>(GmemStrideTypeD{}) == 1) {  // RowMajor FP16
    if constexpr (size<1>(EpilogueTile_MN{}) % 16 == 0) {
        return SM90_U16x8_STSM_T{};   // 选择 stmatrix.x4.trans
    }
}
```

**MMA 输出**：
```
每线程: ((2,2), (4, 8)) = 128 个 values
```

**retile 后**：
```
(8, 16)  // 每次存 8 个 FP16，执行 16 次 stmatrix.x4.trans
```

**数据流示意**：
```
MMA 输出 (每线程视角):
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│16×8 │ │16×8 │ │16×8 │ │16×8 │  RestN=0,1,2,3 (RestM=0)
└─────┘ └─────┘ └─────┘ └─────┘
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│16×8 │ │16×8 │ │16×8 │ │16×8 │  RestM=1
└─────┘ └─────┘ └─────┘ └─────┘

stmatrix.x4.trans 需要:
┌───────────────────────┐
│       16×16           │  一次把 RestN=0,1 合并存储
└───────────────────────┘
```

### 4.2 AutoVectorizing 的 retile

即使使用普通的向量化存储，也需要 retile：

**假设向量化宽度为 4 个 FP16**：
```
retile 前: ((2,2), (4, 8))     // MMA 嵌套布局
retile 后: (4, 32)             // 每次存 4 个元素，执行 32 次
```

**生成的存储代码**：
```cpp
for (int i = 0; i < 32; ++i) {
    // 一次存储 4 个 FP16 = 8 bytes = float2
    *reinterpret_cast<float2*>(&smem[offset]) =
        *reinterpret_cast<float2*>(&reg[i*4]);
}
```

## 5. 常见误解澄清

### 5.1 "retile 是为了解决 rank 不匹配"

**错误**。`cute::copy` 只检查 rank 是否相等：
```cpp
static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");
```

`((2,2), (2,4))` 和 `(8, 4)` 的 rank 都是 2，rank 检查会通过。

**正确理解**：retile 解决的是 **mode 内部的组织方式**问题，确保 values 按照 copy atom 期望的分组方式排列。

### 5.2 "寄存器可以合并访存"

**错误**。寄存器本身不存在"合并访存"的概念——每个线程的寄存器是独立的。

**正确理解**：retile 是为了匹配 **store 指令的操作粒度**。`stmatrix` 指令要求特定数量的寄存器作为输入，retile 确保正确数量的 values 被分组在一起。

### 5.3 "元素总数相同就可以直接 copy"

**错误**。虽然元素总数相同，但 `cute::copy` 的循环模式依赖于 layout 结构：
```cpp
for (int i = 0; i < size<1>(dst_c); ++i) {
    copy_atom.call(src_c(_,i), dst_c(_,i));
}
```

如果第一个 mode 的大小不匹配 copy atom 期望的 values 数，会导致错误的内存访问。

## 6. 总结

| 概念 | 说明 |
|------|------|
| MMA 输出 layout | `((FrgV), (RestM, RestN))` - 嵌套的多维结构 |
| Copy 期望 layout | `(atom_vals, rest_vals)` - 扁平的二维结构 |
| retile 作用 | 重组 values，使其按 copy atom 期望的方式分组 |
| 适用场景 | 所有 r2s/s2r copy，无论使用 stmatrix 还是 AutoVectorizing |

**核心要点**：
1. `retile` 不改变数据，只改变 layout 的表示方式
2. `retile` 确保 `cute::copy` 的循环迭代模式与 copy atom 匹配
3. 无论使用什么 copy 指令，只要 MMA 输出是多维嵌套结构，都需要 retile

## 参考资料

- [CuTe Tiled MMA - Lei Mao's Log Book](https://leimao.github.io/blog/CuTe-Tiled-MMA/)
- [CuTe Tiled Copy - Lei Mao's Log Book](https://leimao.github.io/blog/CuTe-Tiled-Copy/)
- [CUTLASS GitHub Discussion #1142](https://github.com/NVIDIA/cutlass/discussions/1142)
- CUTLASS 源码: `include/cute/atom/copy_atom.hpp`
- CUTLASS 源码: `include/cute/algorithm/copy.hpp`
