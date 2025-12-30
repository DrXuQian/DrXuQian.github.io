---
title: "0x14 TiledCopy 的 tile2thrfrg 详解：Copy 操作的线程分片机制"
date: 2024-12-30 11:00:00
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
  - TiledCopy
  - tile2thrfrg
  - tidfrg_S
  - tidfrg_D
  - ldmatrix
---

本文详细分析 TiledCopy 中的 `tile2thrfrg` 函数，以及 `tidfrg_S` 和 `tidfrg_D` 的工作原理。这些函数实现了 Copy 操作的线程分片，是理解 CUTLASS 数据加载流程的关键。

<!-- more -->

> **与 MMA 的 thrfrg_C 对比**
>
> MMA 的 `thrfrg_C` 见 [0x0F thrfrg_C 详解](/0x0F-thrfrg-c-analysis)。Copy 的 `tile2thrfrg` 原理类似，但多了 **ref2trg 坐标变换**来处理源/目标布局差异。

## 1. 函数入口：tidfrg_S 和 tidfrg_D

### tidfrg_S（Source 分片）

```cpp
template <class STensor>
auto tidfrg_S(STensor&& stensor) {
    return tile2thrfrg(
        zipped_divide(stensor, Tiler_MN{}),
        right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{})
    );
}
```

**读法**：tid-frg-S = Thread ID + Fragment + Source

**含义**：把 Source tensor 按线程分片，返回每个线程负责的 fragment

### tidfrg_D（Destination 分片）

```cpp
template <class DTensor>
auto tidfrg_D(DTensor&& dtensor) {
    return tile2thrfrg(
        zipped_divide(dtensor, Tiler_MN{}),
        right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{})
    );
}
```

**区别**：使用 `AtomLayoutDst` 而不是 `AtomLayoutSrc`

## 2. 核心组件解释

### 2.1 TiledLayout_TV

**TiledCopy 的 (线程, 值) → (M, N) 映射**

```cpp
using TiledLayout_TV = LayoutCopy_TV;  // 来自 TiledCopy 模板参数
```

来源通常是：
- `make_tiled_copy_A(copy_atom, mma)` → `mma.get_layoutA_TV()`
- `make_tiled_copy_B(copy_atom, mma)` → `mma.get_layoutB_TV()`
- `make_tiled_copy(atom, thr_layout, val_layout)` → 自定义

### 2.2 AtomNumThr 和 AtomNumVal

**单个 Copy_Atom 的线程数和每线程值数**

```cpp
using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));
using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));
```

例如 `SM75_U32x4_LDSM_N`（ldmatrix.x4）：
- `AtomNumThr = 32`：需要 32 线程协作
- `AtomNumVal = 4`：每线程获得 4 个 32-bit 寄存器

### 2.3 Tiler_MN

**TiledCopy 覆盖的 tile 大小**

```cpp
using Tiler_MN = ShapeTiler_MN;  // 例如 (64, 32)
```

用于 `zipped_divide(tensor, Tiler_MN)` 把大 tensor 切成小 tile。

### 2.4 AtomLayoutRef, AtomLayoutSrc, AtomLayoutDst

**Copy_Atom 的三个核心 Layout**

```cpp
// 以 SM75_U32x4_LDSM_N 为例
using SrcLayout = Layout<Shape<_32, _128>, Stride<_128, _1>>;      // SMEM 布局
using DstLayout = Layout<Shape<_32, Shape<_32, _4>>, ...>;         // 寄存器布局
using RefLayout = DstLayout;                                        // 参考布局
```

| Layout | 含义 | 用途 |
|--------|------|------|
| **Ref** | 参考布局 | partition 时的统一坐标系 |
| **Src** | Source 布局 | 描述源数据的物理排列 |
| **Dst** | Destination 布局 | 描述目标数据的物理排列 |

## 3. tile2thrfrg 逐步分析

### 源码

```cpp
template <class Tensor, class Ref2TrgLayout>
auto tile2thrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg)
{
    // Step 1: 按 Atom 大小切分 TiledLayout_TV
    auto atom_layout_TV = zipped_divide(TiledLayout_TV{},
                                         make_shape(AtomNumThr{}, AtomNumVal{}));
    // ((atom_tid, atom_val), (rest_tid, rest_val)) -> (m, n)

    // Step 2: 用 ref2trg 变换 Atom 内布局
    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    // ((trg_tid, trg_val), (rest_tid, rest_val)) -> (m, n)

    // Step 3: 重组维度
    auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1, Shape<_1,_1>>{});
    // (TotalThr, (FrgV, FrgX)) -> (m, n)

    // Step 4: 应用到输入 tensor
    auto tv_tensor = tensor.compose(thrval2mn, _);
    // ((thrid, val), (RestM, RestN))

    // Step 5: 展开返回
    return tv_tensor(make_coord(_,_), _);
}
```

### Step 1: zipped_divide 切分

```
TiledLayout_TV: (TiledNumThr, TiledNumVal) -> (M, N)
例如: (128, 8) -> (64, 32)

zipped_divide by (AtomNumThr, AtomNumVal) = (32, 4):

atom_layout_TV: ((32, 4), (4, 2)) -> (64, 32)
                  ↑   ↑    ↑  ↑
             单个Atom  跨Atom
             (thr,val) (thr,val)
```

**图示**：

```
TiledLayout_TV (128 thr × 8 val)
┌─────────────────────────────────────────┐
│   ┌──────────┬──────────┬─────────────┐ │
│   │ Atom 0,0 │ Atom 1,0 │ ... Atom 3,0│ │ ← rest_tid = 4
│   │  32×4    │  32×4    │    32×4     │ │
│   ├──────────┼──────────┼─────────────┤ │
│   │ Atom 0,1 │ Atom 1,1 │ ... Atom 3,1│ │ ← rest_val = 2
│   │  32×4    │  32×4    │    32×4     │ │
│   └──────────┴──────────┴─────────────┘ │
└─────────────────────────────────────────┘
```

### Step 2: compose(ref2trg, _)

```cpp
ref2trg = right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{})
```

**ref2trg 的作用**：把 Ref 坐标系转换到 Src（或 Dst）坐标系

```
Ref: (thr, val) → bit_offset_in_ref
Src: (thr, val) → bit_offset_in_src

ref2trg = Ref⁻¹ ∘ Src
        : bit_offset_in_src → (thr', val')_ref
```

用函数表示 compose：

```
trg_layout_TV(x, y) = atom_layout_TV(ref2trg(x), y)

其中 x = (atom_tid, atom_val)  ← 只变换这部分
     y = (rest_tid, rest_val)  ← 保持不变
```

**为什么需要这个变换？**

因为 ldmatrix 等指令会做 shuffle：
- 线程从 SMEM 读的位置（Src 布局）
- 最终数据在寄存器的位置（Dst/Ref 布局）

两者不同，需要 ref2trg 建立映射。

### Step 3: zip + coalesce

```cpp
auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1, Shape<_1,_1>>{});
```

**zip 重排维度**：

```
输入:  ((atom_tid, atom_val), (rest_tid, rest_val))
输出:  ((atom_tid, rest_tid), (atom_val, rest_val))
     = ((所有线程),           (所有值))
```

**coalesce 合并维度**：

第二个参数 `Shape<_1, Shape<_1,_1>>{}` 是保护形状：
- 第一维 `_1`：线程维度合并成标量
- 第二维 `Shape<_1,_1>`：值维度保持两层结构

```
结果: (TotalThr, (FrgV, FrgX)) -> (m, n)
       ↑           ↑     ↑
    所有线程   Atom内值  跨Atom值
```

**为什么值维度不合并？**

因为一次 Copy_Atom 只能处理 FrgV 个值，需要保留这个信息用于循环迭代：

```cpp
// 实际使用
for (int i = 0; i < FrgX; ++i) {
    copy_atom(src(_, i), dst(_, i));  // 每次处理 FrgV 个值
}
```

### Step 4: tensor.compose(thrval2mn, _)

```cpp
auto tv_tensor = tensor.compose(thrval2mn, _);
```

**函数复合**：

```
tensor: ((TileM, TileN), Rest) → element
thrval2mn: (thr, val) → (m, n)

tv_tensor(x, y) = tensor(thrval2mn(x), y)
                = tensor((m, n), Rest)

其中 x = (thr, val), y = Rest
```

### Step 5: 最终返回

```cpp
return tv_tensor(make_coord(_,_), _);
// 形状: (Thr, Val, Rest...)
```

## 4. 完整例子：ldmatrix 加载 64×32 tile

### 配置

```cpp
// Copy_Atom: SM75_U32x4_LDSM_N (ldmatrix.x4)
AtomNumThr = 32   // 32 线程协作
AtomNumVal = 4    // 每线程 4 个 32-bit 值

// TiledCopy: 128 线程覆盖 64×32 tile
TiledNumThr = 128
TiledNumVal = 8
Tiler_MN = (64, 32)
```

### 流程

```
原始 SMEM tensor: (M, N) = (128, 64)
        ↓ zipped_divide by Tiler_MN = (64, 32)
tensor: ((64, 32), (2, 2))
        ↓ tile2thrfrg
输出: (128, (4, 2), (2, 2))
       ↑     ↑  ↑    ↑
     线程  FrgV FrgX Rest

每个线程负责 4×2×2×2 = 32 个元素
总计: 128 × 32 = 4096 = 128×64/2 ✓
```

### 使用

```cpp
auto thr_tensor = tidfrg_S(smem_tensor);
auto my_tensor = thr_tensor(threadIdx.x, _, _);

// my_tensor 形状: ((4, 2), (2, 2))
// my_tensor(0) 指向当前线程要读的第一个 SMEM 位置
```

## 5. Src vs Dst 布局的差异（ldmatrix 例子）

ldmatrix 的关键特性是**硬件 shuffle**：

```
SMEM (SrcLayout)                  寄存器 (DstLayout/RefLayout)
┌─────────────────┐               ┌────┬────┬────┬────┐
│ T0: 连续128bit  │               │ T0 │ T1 │ T2 │ T3 │
│ T1: 连续128bit  │  ──ldmatrix─→ │reg0│reg0│reg0│reg0│
│ ...             │               │reg1│reg1│reg1│reg1│
└─────────────────┘               └────┴────┴────┴────┘

线程 0 从 SMEM 读 [a,b,c,d]
但最终：T0 持有 [a,e,i,m]（来自不同线程的第一个元素）
```

**ref2trg 变换**确保 partition 正确：给定线程 ID，找到它应该从 SMEM 读取的位置。

## 6. 与 MMA thrfrg_C 的对比

| 方面 | MMA thrfrg_C | Copy tile2thrfrg |
|------|-------------|------------------|
| **用途** | 累加器分片 | 数据加载分片 |
| **布局转换** | 无（直接使用 AtomLayoutC_TV） | 有（ref2trg 变换） |
| **原因** | MMA 输入输出布局一致 | Copy 源/目标布局可能不同 |

## 7. 从嵌套函数角度理解 compose 链

整个 `tile2thrfrg` 可以理解为多层函数嵌套。让我们用数学函数的形式来表示：

### 定义各个函数

```
tensor:         (m, n, rest) → smem_element
TiledLayout_TV: (thr, val) → (m, n)
ref2trg:        (thr, val)_src → (thr', val')_ref
```

### 嵌套过程

**Step 1**: `atom_layout_TV = zipped_divide(TiledLayout_TV, (AtomThr, AtomVal))`

```
atom_layout_TV((atom_thr, atom_val), (rest_thr, rest_val))
  = TiledLayout_TV(atom_thr + AtomThr * rest_thr,
                   atom_val + AtomVal * rest_val)
  = (m, n)
```

**Step 2**: `trg_layout_TV = atom_layout_TV.compose(ref2trg, _)`

```
trg_layout_TV(x, y) = atom_layout_TV(ref2trg(x), y)

展开：
trg_layout_TV((src_thr, src_val), (rest_thr, rest_val))
  = atom_layout_TV(ref2trg(src_thr, src_val), (rest_thr, rest_val))
  = atom_layout_TV((ref_thr, ref_val), (rest_thr, rest_val))
  = (m, n)
```

**Step 3**: `thrval2mn = coalesce(zip(trg_layout_TV), ...)`

```
thrval2mn(thr_flat, val_nested) = (m, n)

其中 thr_flat 是展平后的线程 ID
     val_nested = (atom_val, rest_val) 保持嵌套
```

**Step 4**: `tv_tensor = tensor.compose(thrval2mn, _)`

```
tv_tensor((thr, val), rest) = tensor(thrval2mn(thr, val), rest)
                             = tensor((m, n), rest)
                             = smem_element
```

### 完整的嵌套函数链

将所有 compose 展开，得到最终的函数：

```
tv_tensor(thr_idx, (atom_val, rest_val), rest)
  = tensor(
      thrval2mn(thr_idx, (atom_val, rest_val)),
      rest
    )
  = tensor(
      atom_layout_TV(
        ref2trg(thr_idx % AtomThr, atom_val),
        (thr_idx / AtomThr, rest_val)
      ),
      rest
    )
  = tensor(
      TiledLayout_TV(
        ref_thr + AtomThr * (thr_idx / AtomThr),
        ref_val + AtomVal * rest_val
      ),
      rest
    )
  = tensor((m, n), rest)
  = *(smem_ptr + offset)
```

### 图示

```
输入: (thr_idx, (atom_val, rest_val), rest)
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: 分解 thr_idx                               │
│    atom_thr = thr_idx % AtomThr                     │
│    rest_thr = thr_idx / AtomThr                     │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: ref2trg 变换                               │
│    (ref_thr, ref_val) = ref2trg(atom_thr, atom_val) │
│    把 Src 坐标转换成 Ref 坐标                        │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: TiledLayout_TV 计算                        │
│    (m, n) = TiledLayout_TV(                         │
│               ref_thr + AtomThr * rest_thr,         │
│               ref_val + AtomVal * rest_val          │
│             )                                        │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: tensor 访问                                │
│    element = tensor((m, n), rest)                   │
│            = *(smem_ptr + layout((m,n), rest))      │
└─────────────────────────────────────────────────────┘
              │
              ▼
输出: SMEM element
```

### 核心洞察

**compose 的本质是函数复合**：`A.compose(B)` = `A ∘ B` = `A(B(x))`

整个 `tile2thrfrg` 构建了一个从 `(线程ID, 值ID)` 到 `SMEM 元素` 的映射：

```
(thr_idx, val_idx) ──→ (m, n) ──→ offset ──→ smem_element
                  TiledLayout   tensor.layout
```

**ref2trg 的位置**：插入在中间，负责处理 Src/Dst 布局差异：

```
(src_thr, src_val) ──ref2trg──→ (ref_thr, ref_val) ──TiledLayout──→ (m, n)
```

## 8. 总结

```
tidfrg_S(stensor) 的完整流程：

stensor (M, N)
    ↓ zipped_divide by Tiler_MN
((TileM, TileN), (RestM, RestN))
    ↓ tile2thrfrg with ref2src
        ↓ zipped_divide TiledLayout_TV by (AtomThr, AtomVal)
        ↓ compose with ref2trg（坐标变换）
        ↓ zip + coalesce（重组维度）
        ↓ compose with tensor
(Thr, (FrgV, FrgX), Rest)

最终：每个线程知道自己应该从 SMEM 的哪些位置读取数据
```

## References

- `include/cute/atom/copy_atom.hpp` - TiledCopy 和 tile2thrfrg
- `include/cute/atom/copy_traits_sm75.hpp` - ldmatrix 的 Src/Dst/Ref Layout
- `include/cute/arch/copy_sm75.hpp` - ldmatrix PTX 指令封装
