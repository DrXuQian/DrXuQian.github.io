---
title: "0x16 CuTe TiledCopy retile 函数深度解析"
date: 2024-12-30 14:00:00
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
  - TiledCopy
  - retile
  - Layout
---

本文深入分析 CuTe 中 TiledCopy 的 `retile` 函数，解释其如何将 MMA fragment 布局转换为 Copy Atom 可执行的格式。

<!-- more -->

## 1. retile 的作用

`retile` 函数将已经按某种方式分区的 tensor（如 MMA fragment），重新组织成符合 Copy Atom 执行格式的布局。

### 典型使用场景

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  TiledMMA       │     │    retile       │     │   TiledCopy     │
│  partition_C()  │ ──> │   重新组织布局   │ ──> │   执行 copy     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 实际代码示例

```cpp
// sm80_mma_multistage.hpp
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));  // (MMA,MMA_M,MMA_K) - MMA 布局
Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA); // (CPY,CPY_M,CPY_K) - Copy 布局

// 现在可以执行 copy
copy(smem_tiled_copy_A, tCsA(_,_,_,pipe), tCrA_copy_view);
```

## 2. 源代码

```cpp
// include/cute/atom/copy_atom.hpp
template <class Tensor>
CUTE_HOST_DEVICE constexpr static auto
retile(Tensor&& tensor)
{
  constexpr int R = remove_cvref_t<Tensor>::rank;

  auto V = size<0>(tensor);

  auto frg_layout_mn = upcast<TiledNumThr{} * V>(
      right_inverse(TiledLayout_TV{}).with_shape(shape(Tiler_MN{})));
  // (m,n) -> v_idx

  auto frg_layout_v = zipped_divide(
      logical_product(make_layout(V), right_inverse(frg_layout_mn)),
      make_layout(AtomNumVal{}));
  // (atom_vals,rest_vals) -> (v,m,n)

  // Tile the tensor for TileFrg
  auto t_tensor = zipped_divide(tensor, prepend(product_each(shape(frg_layout_mn)), V));
  // ((TileV,TileM,TileN,...),(1,RestM,RestN,...))

  // Transform the tile mode
  auto v_tensor = t_tensor.compose(frg_layout_v, _);
  // ((atom_vals,rest_vals),(1,RM,RN,...))

  // Unfold and return
  return v_tensor(_, append<R>(Int<0>{},_));
}
```

## 3. 输入输出

| 项目 | 说明 |
|------|------|
| **输入** | `(V, RestM, RestN, ...)` - 某线程的 MMA fragment |
| **输出** | `((AtomVals, RestVals), RestM, RestN, ...)` - 按 Copy Atom 分组 |
| **V** | 输入 tensor 第一维，线程拥有的值数量 |
| **AtomNumVal** | Copy Atom 每次执行需要的值数量 |

## 4. 逐行解析

### 4.1 获取 V

```cpp
auto V = size<0>(tensor);
```

`V` 是输入 tensor 的第一维大小，代表该线程拥有的值数量。通常来自 MMA 的 `partition_fragment_A/B`。

### 4.2 计算 frg_layout_mn

```cpp
auto frg_layout_mn = upcast<TiledNumThr{} * V>(
    right_inverse(TiledLayout_TV{}).with_shape(shape(Tiler_MN{})));
// (m,n) -> v_idx
```

**分解理解**：

1. `TiledLayout_TV{}`: `(thr, val) -> offset` — 整个 tile 的线程-值到偏移的映射
2. `right_inverse(...)`: `offset -> (thr, val)` 的线性索引
3. `.with_shape(Tiler_MN)`: 将输出重塑为 `(M, N)` 坐标形式
4. `upcast<TiledNumThr * V>`: 每 `TiledNumThr * V` 个元素压缩为一组

**upcast 的效果**：

```
upcast<N>(layout) 同时影响输入和输出：
- 输入 shape：缩小 N 倍
- 输出 stride：缩小 N 倍

结果：(m', n') -> v_group_idx
      其中 (m', n') 是单线程的坐标范围
```

**图解**：

```
假设 TiledNumThr=4, V=4, Tiler_MN=(4,8)

upcast 之前: (m,n) -> linear_idx [0, 32)

      N →
      0   1   2   3   4   5   6   7
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
  0 │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
M ├───┼───┼───┼───┼───┼───┼───┼───┤
↓ 1 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │
    ├───┼───┼───┼───┼───┼───┼───┼───┤
  2 │16 │17 │18 │19 │20 │21 │22 │23 │
    ├───┼───┼───┼───┼───┼───┼───┼───┤
  3 │24 │25 │26 │27 │28 │29 │30 │31 │
    └───┴───┴───┴───┴───┴───┴───┴───┘

upcast<16> 之后: (m,n) -> linear_idx / 16 = v_group_idx

      N →
      0   1   2   3   4   5   6   7
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
  0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │  ← v_group 0
M ├───┼───┼───┼───┼───┼───┼───┼───┤
↓ 1 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │  ← v_group 0
    ├───┼───┼───┼───┼───┼───┼───┼───┤
  2 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │  ← v_group 1
    ├───┼───┼───┼───┼───┼───┼───┼───┤
  3 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │  ← v_group 1
    └───┴───┴───┴───┴───┴───┴───┴───┘
```

**关键点**：`frg_layout_mn` 是**多对一映射**（非单射），多个 `(m,n)` 映射到同一个 `v_group_idx`。

### 4.3 计算 frg_layout_v

```cpp
auto frg_layout_v = zipped_divide(
    logical_product(make_layout(V), right_inverse(frg_layout_mn)),
    make_layout(AtomNumVal{}));
// (atom_vals, rest_vals) -> (v, m, n)
```

**分解理解**：

1. `right_inverse(frg_layout_mn)`: 由于 `frg_layout_mn` 非单射，返回的是形状信息
2. `logical_product(make_layout(V), ...)`: 将 V 个值与 v_group 形状组合
3. `zipped_divide(..., AtomNumVal)`: 按 Copy Atom 需要的值数量分组

**zipped_divide 的目的**：

```
V 和 AtomNumVal 通常不同：
- V = 8 (MMA fragment 有 8 个值)
- AtomNumVal = 2 (Copy Atom 每次处理 2 个值)

zipped_divide 后：
- atom_vals: [0, AtomNumVal) = [0, 2) — 一次 copy 的值
- rest_vals: [0, V/AtomNumVal) = [0, 4) — 需要 4 次 copy
```

### 4.4 Tile tensor

```cpp
auto t_tensor = zipped_divide(tensor, prepend(product_each(shape(frg_layout_mn)), V));
// ((TileV,TileM,TileN,...),(1,RestM,RestN,...))
```

将输入 tensor 按 `frg_layout_mn` 的形状进行 tile 划分。

### 4.5 应用布局变换

```cpp
auto v_tensor = t_tensor.compose(frg_layout_v, _);
// ((atom_vals,rest_vals),(1,RM,RN,...))
```

通过 `compose` 应用 `frg_layout_v` 变换，重新组织值的排列。

### 4.6 保持 rank 一致

```cpp
return v_tensor(_, append<R>(Int<0>{},_));
```

**作用**：保持输出 rank 与输入一致。

```
输入 tensor: (V, M, K) → rank = 3
中间 v_tensor: ((atom_vals, rest_vals), (1, RestM, RestK)) → rank = 2（嵌套）
输出: ((atom_vals, rest_vals), RestM, RestK) → rank = 3

append<R>(Int<0>{}, _) 生成 (0, _, _, ...) 共 R 个元素：
- 0：取第二维的第一个元素（那个 1），消掉它
- _：保留后面的 RestM, RestK
```

## 5. 完整例子

### 配置

```cpp
TiledNumThr = 4      // 4 个线程
TiledNumVal = 8      // 每线程 8 个值
V = 4                // 输入 tensor 第一维 (MMA fragment)
AtomNumVal = 2       // Copy Atom 每次处理 2 个值
Tiler_MN = (4, 8)    // Tile 大小
```

### 计算过程

```
Step 1: frg_layout_mn
  upcast<16>(right_inverse(TiledLayout_TV).with_shape((4,8)))
  结果: (m,n) -> v_group_idx, 共 2 个 v_group

Step 2: frg_layout_v
  logical_product(make_layout(4), right_inverse(frg_layout_mn))
  → (v_idx, v_group_idx) = (4, 2) 共 8 个值

  zipped_divide by AtomNumVal=2
  → (atom_vals, rest_vals) = (2, 4)
     每次 copy 2 个值，执行 4 次

Step 3: 应用到 tensor
  输入: (4, RestM, RestN)
  输出: ((2, 4), RestM, RestN) = (CPY, CPY_M, CPY_N)
```

### 图解

```
输入 tensor (V=4, ...):
┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │  线程的 4 个值
└───┴───┴───┴───┘

经过 retile 后 ((AtomVals=2, RestVals=2), ...):
┌─────────┬─────────┐
│  0, 1   │  2, 3   │
│ (copy0) │ (copy1) │  2 次 copy，每次 2 个值
└─────────┴─────────┘

实际情况 V=8, AtomNumVal=2:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
└───┴───┴───┴───┴───┴───┴───┴───┘
  ├───┤   ├───┤   ├───┤   ├───┤
  copy0   copy1   copy2   copy3

输出: ((2, 4), RestM, RestN)
```

## 6. 与 partition_S/D 的区别

| 函数 | 输入 | 用途 |
|------|------|------|
| `partition_S/D` | SMEM tensor | 将 SMEM 按 TiledCopy 布局分区 |
| `retile_S/D` | 已分区的 fragment | 将其他布局转为 Copy 布局 |

```cpp
// partition_S: SMEM -> Copy 布局
Tensor tCsA = smem_thr_copy.partition_S(sA);  // 从 SMEM 分区

// retile_D: MMA 布局 -> Copy 布局
Tensor tCrA_view = smem_thr_copy.retile_D(tCrA);  // 重组 MMA fragment

// 现在两者布局兼容，可以 copy
copy(smem_tiled_copy, tCsA, tCrA_view);
```

## 7. 总结

```
┌─────────────────────────────────────────────────────────────────┐
│                        retile 函数                              │
├─────────────────────────────────────────────────────────────────┤
│  目的: 将任意分区的 tensor 重组为 Copy Atom 可执行格式           │
├─────────────────────────────────────────────────────────────────┤
│  输入: (V, Rest...) — 某线程的 fragment                         │
│  输出: ((AtomVals, RestVals), Rest...) — 按 copy 指令分组        │
├─────────────────────────────────────────────────────────────────┤
│  核心步骤:                                                       │
│  1. frg_layout_mn: 计算值在 (m,n) 空间的分组 (多对一映射)        │
│  2. frg_layout_v: 按 AtomNumVal 重新分组值                       │
│  3. compose: 应用变换到 tensor                                   │
│  4. 展开: 保持输出 rank 与输入一致                               │
├─────────────────────────────────────────────────────────────────┤
│  关键参数:                                                       │
│  - V: 输入第一维大小 (来自 MMA 等)                               │
│  - AtomNumVal: Copy Atom 每次需要的值数量                        │
│  - TiledNumThr: TiledCopy 的线程数                               │
└─────────────────────────────────────────────────────────────────┘
```

## References

- `include/cute/atom/copy_atom.hpp` - retile 函数定义
- `include/cutlass/gemm/collective/sm80_mma_multistage.hpp` - retile_D 使用示例
- `include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` - retile_S 使用示例
