---
title: "0x11 CuTe TiledMMA 完全指南：从 Layout 到线程映射"
date: 2024-12-24 17:00:00
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
  - TiledMMA
  - Layout
  - thrfrg_C
  - get_slice
---

本文是 CuTe TiledMMA 的综合指南，将核心概念串联起来，深入剖析从 Layout 基础到线程映射的完整流程。

<!-- more -->

> **相关文章**:
> - [0x01 TV Layout Guide](/2024/12/24/0x01-tv-layout-guide/) - TV Layout 基础
> - [0x05 make_tiled_mma](/2024/12/24/0x05-make-tiled-mma/) - TiledMMA 构建
> - [0x0E Layout 分割操作](/2024/12/24/0x0E-cute-layout-divide/) - logical_divide 与 zipped_divide
> - [0x0F thrfrg_C 详解](/2024/12/24/0x0F-thrfrg-c-analysis/) - C Tensor 线程分片

## 1. 核心概念速查表

| 概念 | 本质 |
|------|------|
| **Layout** | 坐标到 offset 的映射函数 |
| **Tiler** | 坐标空间的分解规则 |
| **logical_divide** | 按 (inner, outer) 重组坐标层级 |
| **zipped_divide** | 按 (M方向, N方向) 重组坐标层级 |
| **compose** | 两个映射的复合：coord → offset₁ → offset₂ |
| **Shape** | 坐标的取值范围 |
| **Stride** | 每个坐标维度对 offset 的贡献系数 |

### 概念关系图

```
                    Tiler (分解规则)
                         │
                         ▼
原始坐标 ──────────────────────────────► 分层坐标
 (m, n)                                ((m_i,m_o), (n_i,n_o))  [zipped]
                                       ((m_i,n_i), (m_o,n_o))  [logical]
    │                                         │
    │ Layout                                  │ Layout'
    ▼                                         ▼
 offset ◄──────────────────────────────── offset
                   (相同)
```

## 2. Layout 基础

Layout 本质是一个**坐标到整数偏移量的映射函数**：

$$L: \text{coord} \rightarrow \text{offset}$$

### CuTe 表示法

```
Layout = Shape : Stride
```

```cpp
auto layout = make_layout(make_shape(8, 4), make_stride(1, 8));
// (8, 4) : (1, 8)
// L(m, n) = m × 1 + n × 8
```

> 详细的 TV Layout 解析请参考 [0x01 TV Layout Guide](/2024/12/24/0x01-tv-layout-guide/)

## 3. Divide 操作

关于 `logical_divide` 和 `zipped_divide` 的详细解析，请参考 [0x0E Layout 分割操作](/2024/12/24/0x0E-cute-layout-divide/)。

这里给出核心对比：

| Operation | Shape | Stride | 特点 |
|-----------|-------|--------|------|
| logical_divide | `((_4,_2),(_2,_4))` | `((_1,_4),(_8,_16))` | 保持原始 offset |
| zipped_divide | `((_4,_2),(_2,_4))` | `((_1,_8),(_4,_16))` | Tile 内连续 |

**关键理解：divide 操作后仍然是 Layout！** 只是 Shape 变成了嵌套结构。

## 4. TiledMMA 结构

### 核心组成

```cpp
template <class MMA_Atom, class AtomLayoutMNK, class PermutationMNK>
struct TiledMMA : MMA_Atom
{
  // 从 MMA_Atom 继承
  using AtomShape_MNK  = typename MMA_Atom::Shape_MNK;    // 如 (64, 16, 16)
  using AtomThrID      = typename MMA_Atom::ThrID;        // 如 Layout<_128>
  using AtomLayoutC_TV = typename MMA_Atom::LayoutC_TV;   // Atom 内的 TV 映射

  // TiledMMA 自己的成员
  using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
  ThrLayoutVMNK thr_layout_vmnk_;  // 核心：线程的 (V,M,N,K) 布局
};
```

### 层级关系

```
TiledMMA
├── MMA_Atom (单个 MMA 指令)
│   ├── AtomShape_MNK: (64, 16, 16)     -- 单个 Atom 覆盖的区域
│   ├── AtomThrID: 128                  -- Atom 内的线程数
│   └── AtomLayoutC_TV: CLayout_64x16   -- Atom 内 (thr,val)→offset
│
├── AtomLayoutMNK: (2, 1, 1)            -- Atom 的 tiling 方式
│   └── M方向2个Atom, N方向1个, K方向1个
│
└── thr_layout_vmnk_: (128, 2, 1, 1)    -- 全局线程布局
    └── 128×2×1×1 = 256 总线程
```

> 关于 TiledMMA 的构建，请参考 [0x05 make_tiled_mma](/2024/12/24/0x05-make-tiled-mma/)

## 5. thrfrg_C 函数

`thrfrg_C` 是 TiledMMA 的核心函数，将 C 矩阵 Layout 转换为线程-值视图。

> 详细的 thrfrg_C 解析请参考 [0x0F thrfrg_C 详解](/2024/12/24/0x0F-thrfrg-c-analysis/)

### 流程概览

```
ctensor: (128, 128)
    │
    ▼ zipped_divide by (64, 16)

c_tensor: ((64, 16), (2, 8))
          ((AtomM,AtomN), (RestM,RestN))
    │
    ▼ compose with AtomLayoutC_TV

tv_tensor: ((128, 8), (2, 8))
           ((ThrV,FrgV), (RestM,RestN))
    │
    ▼ zipped_divide by (_, (2, 1))

thr_tensor: ((128, (2, 1)), (8, (1, 8)))
            ((ThrV,(ThrM,ThrN)), (FrgV,(RestM',RestN')))
```

## 6. get_slice 与线程映射

### 函数实现

```cpp
template <class ThrIdx>
auto get_slice(ThrIdx const& thr_idx) const
{
    // Step 1: 线性 thread_id → 多维坐标 (v, m, n, k)
    auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);

    // Step 2: 返回该线程的 MMA slice
    return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
}
```

### get_flat_coord 详解

**本质：Layout 的逆映射，从 offset 反推坐标**

```
正向: coord → offset    (Layout 正常用法)
逆向: offset → coord    (get_flat_coord)
```

#### 示例

```cpp
// thr_layout_vmnk_ = (128, 2, 1, 1) : (1, 128, 256, 256)

thr_idx = 0   → thr_vmnk = (0,   0, 0, 0)
thr_idx = 1   → thr_vmnk = (1,   0, 0, 0)
thr_idx = 127 → thr_vmnk = (127, 0, 0, 0)
thr_idx = 128 → thr_vmnk = (0,   1, 0, 0)   // 第二个 Atom
thr_idx = 129 → thr_vmnk = (1,   1, 0, 0)
thr_idx = 255 → thr_vmnk = (127, 1, 0, 0)
```

#### 计算公式

```cpp
// Layout: (128, 2, 1, 1) : (1, 128, 256, 256)
// thr_idx = 130

v = (130 / 1)   % 128 = 2
m = (130 / 128) % 2   = 1
n = (130 / 256) % 1   = 0
k = (130 / 256) % 1   = 0

→ thr_vmnk = (2, 1, 0, 0)
```

### 线程映射图示

```
thr_idx (线性)                    thr_vmnk (多维坐标)
     │                                  │
     │   get_flat_coord                 │
     ▼                                  ▼
   130  ──────────────────────►  (v=2, m=1, n=0, k=0)
                                   │
                                   │ 含义
                                   ▼
                            Atom 内第 2 号线程
                            第 1 个 Atom (M方向)
```

### ThrMMA 的作用

```cpp
template <class TiledMMA, class ThrVMNK>
struct ThrMMA {
    TiledMMA mma_;
    ThrVMNK  thr_vmnk_;  // (v, m, n, k) 坐标

    // 获取该线程负责的 C 数据
    template <class CTensor>
    auto partition_C(CTensor&& ctensor) const {
        auto thr_tensor = mma_.thrfrg_C(ctensor);
        // thr_tensor: ((ThrV, (ThrM, ThrN)), (FrgV, (RestM, RestN)))

        // 用 thr_vmnk_ 选择该线程的 slice
        return thr_tensor(thr_vmnk_, _);
        // 结果: (FrgV, (RestM, RestN)) - 只属于这个线程的数据
    }
};
```

## 7. get_layoutC_TV 解析

### 函数目的

返回一个 Layout，映射 **(thr_idx, val_idx) → C 矩阵中的 offset**

### 实现代码

```cpp
auto get_layoutC_TV() const
{
    // Step 1: 构造参考 C 矩阵 Layout
    auto ref_C = make_layout(make_shape(tile_size_mnk<0>(), tile_size_mnk<1>()));
    // 例如: (128, 128) : (1, 128)

    // Step 2: 构造 thr_idx → 层级坐标 的转换
    auto thridx_2_thrid = composition(
        make_layout(make_shape (size(thr_layout_vmnk_), Int<1>{}),
                    make_stride(Int<1>{},               Int<0>{})),
        right_inverse(make_layout(thr_layout_vmnk_, complement(thr_layout_vmnk_)))
    );

    // Step 3: compose 得到最终 Layout
    return thrfrg_C(ref_C).compose(thridx_2_thrid, _);
}
```

### 流程图

```
输入: (thr_idx, val_idx)
         │
         ▼
    ┌─────────────────┐
    │  thridx_2_thrid │  thr_idx → (ThrV, (ThrM, ThrN))
    └─────────────────┘
         │
         ▼
((ThrV, (ThrM, ThrN)), val_idx)
         │
         ▼
    ┌─────────────────┐
    │  thrfrg_C(ref_C)│  线程坐标 + 值坐标 → offset
    └─────────────────┘
         │
         ▼
输出: offset (在 M×N 矩阵中的位置)
```

### AtomLayoutC_TV vs get_layoutC_TV()

| 属性 | AtomLayoutC_TV | get_layoutC_TV() |
|------|---------------|------------------|
| **级别** | Atom | TiledMMA (整个 Tile) |
| **线程范围** | Atom 内 (128) | 全部线程 (256) |
| **值范围** | 单 Atom 内 (8) | 跨所有 Atom (64) |
| **输出 offset** | Atom 内 offset | Tile 内 offset |
| **覆盖区域** | 64×16 | 128×128 |

### 使用示例

```cpp
auto layoutC_TV = tiled_mma.get_layoutC_TV();

// Thread 130 的 Value 3 在 C 矩阵中的位置
int offset = layoutC_TV(130, 3);
int m = offset % M;
int n = offset / M;
// 现在知道 Thread 130 的第 3 个值对应 C[m, n]
```

## 8. 完整示例代码

```cpp
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

// CLayout_64x16 定义
template <int N>
using CLayout_64xN = Layout<Shape <Shape <  _4, _8, _4>, Shape < _2, _2, Int<N/8>>>,
                            Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;
using CLayout_64x16 = CLayout_64xN<16>;

int main() {
    // 配置
    using AtomShapeM = Int<64>;
    using AtomShapeN = Int<16>;
    using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;  // M方向2个Atom
    using AtomLayoutC_TV = CLayout_64x16;

    // 原始 C 矩阵
    auto ctensor = make_layout(make_shape(Int<128>{}, Int<128>{}));

    // Step 2: zipped_divide by AtomShape
    auto c_tile = make_tile(make_layout(AtomShapeM{}), make_layout(AtomShapeN{}));
    auto c_tensor = zipped_divide(ctensor, c_tile);
    // ((64, 16), (2, 8)) : ((1, 128), (64, 2048))

    // Step 3: compose with AtomLayoutC_TV
    auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{}, _);
    // ((128, 8), (2, 8)) : (...)

    // Step 4: zipped_divide for Thread Tiling
    auto AtomThrID = Layout<Int<128>>{};
    auto thr_layout_vmnk = tiled_product(AtomThrID, AtomLayoutMNK{});
    // (128, 2, 1, 1) : (1, 128, 256, 256)

    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<2>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
    // ((128, (2, 1)), (8, (1, 8)))

    // 验证
    int ThrV = size<0,0>(thr_tensor);   // 128
    int ThrM = size<0,1,0>(thr_tensor); // 2
    int ThrN = size<0,1,1>(thr_tensor); // 1
    int FrgV = size<1,0>(thr_tensor);   // 8
    int RestM = size<1,1,0>(thr_tensor);// 1
    int RestN = size<1,1,1>(thr_tensor);// 8

    // 总线程: 128 × 2 × 1 = 256
    // 每线程值: 8 × 1 × 8 = 64
    // 总元素: 256 × 64 = 16384 = 128 × 128 ✓

    return 0;
}
```

## 9. 总结

### 核心理解

1. **Layout** 是坐标到 offset 的映射，divide 操作只是重组坐标层级，最终 offset 不变

2. **TiledMMA** 通过 thrfrg_C/A/B 将矩阵 Layout 转换为 ((线程坐标), (值坐标)) 的形式

3. **thrfrg_C** 的四个步骤：
   - Permutation（可选重排）
   - 按 Atom 大小切分
   - (M,N) → (Thread, Value) 变换
   - 分配 Atom 给不同 Warpgroup

4. **get_slice** 通过逆映射把线性 thread_id 转为多维坐标，返回线程视图

5. **AtomLayoutC_TV** 是 Atom 级别的映射，**get_layoutC_TV()** 是整个 Tile 级别的映射

### 一图总结

```
C 矩阵 (M×N)
     │
     │ thrfrg_C
     ▼
((ThrV, (ThrM, ThrN)), (FrgV, (RestM, RestN)))
   └────── 线程坐标 ──────┘  └──── 值坐标 ────┘
           │                        │
           │ get_slice(thr_idx)     │
           ▼                        ▼
    特定线程的坐标              该线程负责的所有值
```

## References

- [0x01 TV Layout Guide](/2024/12/24/0x01-tv-layout-guide/)
- [0x05 make_tiled_mma](/2024/12/24/0x05-make-tiled-mma/)
- [0x0E Layout 分割操作](/2024/12/24/0x0E-cute-layout-divide/)
- [0x0F thrfrg_C 详解](/2024/12/24/0x0F-thrfrg-c-analysis/)
- [CUTLASS CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
