---
title: "0x0F CuTe thrfrg_C 详解：C Tensor 的线程分片机制"
date: 2024-12-24
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
  - thrfrg_C
  - TiledMMA
  - Thread-Value Layout
---

本文逐步分析 `thrfrg_C`，即 CUTLASS TiledMMA 中 C（累加器）Tensor 的线程分片函数。理解这一机制对于掌握 WGMMA 如何在线程间分配输出矩阵至关重要。

<!-- more -->

> **示例代码**:
> - [0x0F_thrfrg_C.cu](https://github.com/DrXuQian/cute-examples/blob/main/0x0F_thrfrg_C.cu)
> - [0x0F_thrfrg_C_latex.cu](https://github.com/DrXuQian/cute-examples/blob/main/0x0F_thrfrg_C_latex.cu) (LaTeX 可视化生成)

> **Key Points**
> 1. **thrfrg_C** transforms tensor → thread-indexed fragment view
> 2. **CLayout_64x16**: 128 threads × 8 values = 1024 elements (64×16 Atom)
> 3. **zipped_divide + compose**: Core layout transformation operations
> 4. **AtomLayoutMNK**: Enables cooperative tiling across warpgroups
> 5. **Final shape**: `((ThrV, (ThrM, ThrN)), (FrgV, (RestM, RestN)))`

## 1. Configuration

```cpp
// Single Atom size: 64 rows × 16 columns
using AtomShapeM = Int<64>;
using AtomShapeN = Int<16>;

// Cooperative tiling: 2 Atoms in M direction
using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;

// Thread-Value layout for single Atom (128 threads × 8 values)
using AtomLayoutC_TV = CLayout_64x16;
```

### CLayout_64x16 Definition

```cpp
template <int N>
using CLayout_64xN = Layout<
    Shape <Shape <  _4, _8, _4>, Shape < _2, _2, Int<N/8>>>,
    Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;

using CLayout_64x16 = CLayout_64xN<16>;
// Shape:  ((_4,_8,_4),(_2,_2,_2))
// Stride: ((_128,_1,_16),(_64,_8,_512))
// = 128 threads × 8 values = 1024 elements
```

**Visualization**: [step3_atom_layout_tv.pdf](/assets/cute-layout-divide/step3_atom_layout_tv.pdf)

## 2. Step 0: Original C Tensor

```cpp
auto ctensor = make_layout(make_shape(Int<128>{}, Int<128>{}));
// ctensor: (_128,_128):(_1,_128)
// Total: 16384 elements
```

A 128×128 output matrix in row-major layout.

## 3. Step 1: Permutation (Identity)

```cpp
auto t_tensor = ctensor;
// No change: (_128,_128):(_1,_128)
```

In this example, no permutation is applied. In general, permutation rearranges tensor dimensions.

## 4. Step 2: zipped_divide by AtomShape

```cpp
auto c_tile = make_tile(make_layout(AtomShapeM{}),   // M: 64
                        make_layout(AtomShapeN{}));  // N: 16
auto c_tensor = zipped_divide(t_tensor, c_tile);
// Result: ((_64,_16),(_2,_8)):((_1,_128),(_64,_2048))
```

**Visualization**: [step6_zipped_divide_16x16.pdf](/assets/cute-layout-divide/step6_zipped_divide_16x16.pdf)

### Shape Decomposition

| Dimension | Components | Meaning |
|-----------|------------|---------|
| First `(_64,_16)` | Atom interior | 64 rows × 16 cols per Atom |
| Second `(_2,_8)` | Atom indices | 2×8 = 16 Atoms total |

### Interpretation

- 128×128 matrix divided into 64×16 Atoms
- M direction: 128/64 = 2 Atoms
- N direction: 128/16 = 8 Atoms
- Each Atom: 1024 elements

## 5. Step 3: compose with AtomLayoutC_TV

```cpp
auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{}, _);
// Result: Complex TV layout with thread and value decomposition
```

This is the critical step! `compose` replaces the Atom-interior layout with the Thread-Value layout.

### AtomLayoutC_TV Analysis

```
AtomLayoutC_TV: ((_4,_8,_4),(_2,_2,_2)):((_128,_1,_16),(_64,_8,_512))
  Thr shape: (_4,_8,_4) = 128 threads
  Val shape: (_2,_2,_2) = 8 values/thread
  Total: 128 × 8 = 1024 elements
```

**Thread 0's Values**: [step5_thread0_positions.pdf](/assets/cute-layout-divide/step5_thread0_positions.pdf)

### Thread-Value Layout Visualization

```
(128 threads × 8 values):
         0      1      2      3      4      5      6      7
    +------+------+------+------+------+------+------+------+
 0  |    0 |   64 |    8 |   72 |  512 |  576 |  520 |  584 |
    +------+------+------+------+------+------+------+------+
 1  |  128 |  192 |  136 |  200 |  640 |  704 |  648 |  712 |
    +------+------+------+------+------+------+------+------+
 2  |  256 |  320 |  264 |  328 |  768 |  832 |  776 |  840 |
    +------+------+------+------+------+------+------+------+
...
```

**Full layout**: [step4_clayout_64x16.pdf](/assets/cute-layout-divide/step4_clayout_64x16.pdf)

### Thread 0's 8 Values

```
Val_idx -> offset -> (m, n)
V0 -> offset   0 -> (m= 0, n= 0)
V1 -> offset  64 -> (m= 0, n= 1)
V2 -> offset   8 -> (m= 8, n= 0)
V3 -> offset  72 -> (m= 8, n= 1)
V4 -> offset 512 -> (m= 0, n= 8)
V5 -> offset 576 -> (m= 0, n= 9)
V6 -> offset 520 -> (m= 8, n= 8)
V7 -> offset 584 -> (m= 8, n= 9)
```

Pattern: Thread 0 holds 2×2 blocks at positions (0,0), (0,8), (8,0), (8,8) in the 64×16 Atom.

## 6. Step 4: zipped_divide for Thread Tiling

```cpp
auto AtomThrID = Layout<Int<128>>{};
auto thr_layout_vmnk = tiled_product(AtomThrID, AtomLayoutMNK{});
// thr_layout_vmnk: (_128,_2,_1,_1):(_1,_128,_0,_0)
// Total threads: 256

auto thr_tile = make_tile(_,
                          make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                    make_layout(size<2>(thr_layout_vmnk))));
// thr_tile: (_, (_2:_1, _1:_0))

auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
```

### tiled_product Analysis

`tiled_product(AtomThrID, AtomLayoutMNK)`:
- AtomThrID: 128 threads per Atom
- AtomLayoutMNK: 2×1×1 Atoms (cooperative in M direction)
- Result: 128×2×1×1 = 256 total threads

### Final Result

```
thr_tensor: (((_4,_8,_4),(_2,_1)),((_2,_2,_2),(_1,_8)))
Shape:      ((ThrV, (ThrM, ThrN)), (FrgV, (RestM', RestN')))
```

| Component | Value | Meaning |
|-----------|-------|---------|
| ThrV | 128 | Threads per Atom |
| ThrM | 2 | Atoms in M (cooperative) |
| ThrN | 1 | Atoms in N |
| FrgV | 8 | Values per thread per Atom |
| RestM' | 1 | M tiles remaining per thread |
| RestN' | 8 | N tiles remaining per thread |

## 7. Summary Verification

```
Thread coordinates:
  ThrV  = 128 (threads per Atom)
  ThrM  = 2   (Atoms in M direction)
  ThrN  = 1   (Atoms in N direction)
  Total = 128 × 2 × 1 = 256 threads

Per-thread values:
  FrgV   = 8 (values per thread per Atom)
  RestM' = 1
  RestN' = 8
  Values per thread = 8 × 1 × 8 = 64

Verification:
  Total = threads × values_per_thread
        = 256 × 64 = 16384
  Expected = 128 × 128 = 16384  ✓
```

## 8. Thread Data Examples

### Thread 0 (ThrV=0, ThrM=0, ThrN=0)

- Belongs to: Atom 0 (Warpgroup 0)
- Covers M rows: 0-63
- Values: 64 elements across 8 N-tiles

### Thread 128 (ThrV=0, ThrM=1, ThrN=0)

- Belongs to: Atom 1 (Warpgroup 1)
- Covers M rows: 64-127
- Same value pattern but offset by 64 rows

### Comparing First 4 Threads

```
Thread | V0 offset (m,n) | V1 offset (m,n) | V2 offset (m,n) | V3 offset (m,n)
-------|-----------------|-----------------|-----------------|----------------
   0   |    0 ( 0, 0)    |   64 ( 0, 1)    |    8 ( 8, 0)    |   72 ( 8, 1)
   1   |  128 ( 0, 2)    |  192 ( 0, 3)    |  136 ( 8, 2)    |  200 ( 8, 3)
   2   |  256 ( 0, 4)    |  320 ( 0, 5)    |  264 ( 8, 4)    |  328 ( 8, 5)
   3   |  384 ( 0, 6)    |  448 ( 0, 7)    |  392 ( 8, 6)    |  456 ( 8, 7)
```

Pattern: Adjacent threads handle adjacent N columns within the same M rows.

## 9. Data Flow Diagram

```
Step 0: ctensor (128,128)
            |
            v
Step 2: zipped_divide by AtomShape (64,16)
            |
            | Shape: ((_64,_16), (_2,_8))
            |        Atom内部    Atom索引
            v
Step 3: compose with CLayout_64x16
            |
            | Shape: ((128threads, 8values), (2,8))
            |        Thread-Value    Atom索引
            v
Step 4: zipped_divide by Thread Tiling
            |
            | Shape: ((ThrV,(ThrM,ThrN)), (FrgV,(RestM,RestN)))
            |        Thread坐标        Per-thread数据
            v
Final: thr_tensor for per-thread access
```

## 10. Practical Usage

In actual CUTLASS code, `thrfrg_C` is used to:

1. **Create fragment view**: Each thread knows which elements to accumulate
2. **Enable WGMMA**: Hardware MMA operations expect specific data layouts
3. **Cooperative tiling**: Multiple warpgroups collaborate on larger tiles

```cpp
// In TiledMMA::thrfrg_C
auto thr_tensor = thrfrg_C(ctensor);

// Thread-specific slice
auto thr_frg = thr_tensor(_, threadIdx.x, _);
// This gives the 64 elements this thread is responsible for
```

## 11. PDF Visualizations

All visualizations generated by `print_latex`:

1. [step0_original_8x8.pdf](/assets/cute-layout-divide/step0_original_8x8.pdf) - Original layout
2. [step1_logical_divide.pdf](/assets/cute-layout-divide/step1_logical_divide.pdf) - logical_divide result
3. [step2_zipped_divide.pdf](/assets/cute-layout-divide/step2_zipped_divide.pdf) - zipped_divide result
4. [step3_atom_layout_tv.pdf](/assets/cute-layout-divide/step3_atom_layout_tv.pdf) - AtomLayoutC_TV
5. [step4_clayout_64x16.pdf](/assets/cute-layout-divide/step4_clayout_64x16.pdf) - CLayout_64x16 (32 threads)
6. [step5_thread0_positions.pdf](/assets/cute-layout-divide/step5_thread0_positions.pdf) - Thread 0's values
7. [step6_zipped_divide_16x16.pdf](/assets/cute-layout-divide/step6_zipped_divide_16x16.pdf) - 16x16 zipped_divide

## References

- Source: `cute-examples/thrfrg_C.cu`
- Source: `cute-examples/zipped_divide.cu`
- CUTLASS: `include/cute/atom/mma_atom.hpp`
