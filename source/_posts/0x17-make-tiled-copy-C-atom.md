---
title: "0x17 CuTe make_tiled_copy_C_atom 深度解析"
date: 2024-12-30 16:00:00
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
  - TiledCopy
  - MMA
  - Epilogue
---

本文深入分析 CuTe 中 `make_tiled_copy_C_atom` 函数，解释它如何从 MMA 累加器布局创建一个匹配的 TiledCopy，作为 epilogue 阶段写出数据的桥梁。

<!-- more -->

## 1. 函数作用

`make_tiled_copy_C_atom` 创建一个 TiledCopy，其 `(tid, vid)` 布局与 MMA 累加器的布局匹配。这个 TiledCopy 通常不直接使用，而是作为"参考布局"来创建实际执行的 TiledCopy。

### 典型使用场景

```cpp
// Epilogue: 将 MMA 累加器写到 SMEM/GMEM
// 1. 创建匹配 MMA 布局的"桥接" TiledCopy
TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, tiled_mma);

// 2. 用它作为参考，创建实际执行的 TiledCopy
TiledCopy tiled_r2s = make_tiled_copy_S(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_copy_C_atom);

// 3. 执行 copy
Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);  // 重组寄存器布局
Tensor tRS_sD   = thread_r2s.partition_D(sD_epi);     // 分区 SMEM
copy(tiled_r2s, tRS_rAcc, tRS_sD);
```

## 2. 源代码

```cpp
// include/cute/atom/copy_atom.hpp
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE auto
make_tiled_copy_C_atom(Copy_Atom<CArgs...> const& copy_atom,
                       TiledMMA<MArgs...>  const& mma)
{
  // Truncate the V-layout to just the Copy_Atom, keep the V-order
  auto layoutC_TV = mma.get_layoutC_TV();
  auto copy_V     = Int<Copy_Atom<CArgs...>::NumValSrc>{};
  CUTE_STATIC_ASSERT_V(copy_V <= size<1>(layoutC_TV));
  auto layout_TV  = composition(layoutC_TV, make_layout(make_shape(size<0>(layoutC_TV), copy_V)));

  // Tiler -- Find the active elements in the MMA tensor and generate a tiler
  auto mma_tiler = make_shape(tile_size<0>(mma), tile_size<1>(mma));
  auto mma_zeros = repeat_like(mma_tiler, Int<0>{});

  auto tiler = transform(make_seq<rank(mma_tiler)>{}, [&](auto i) {
    return filter(composition(make_layout(mma_tiler, replace<i>(mma_zeros, Int<1>{})), layout_TV));
  });

  // Layout_TV -- Find the (tid,vid) -> tile coord transformation
  auto tile2mma = composition(make_layout(mma_tiler), tiler);
  auto layout_tv = composition(left_inverse(tile2mma), layout_TV);

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}
```

## 3. 核心输出：layout_tv

`layout_tv` 是这个函数最重要的输出，它是 MMA 布局到 Copy 布局的"翻译表"：

| 侧 | 含义 |
|---|------|
| **输入 `(tid, vid)`** | MMA 累加器布局：线程 tid 的第 vid 个寄存器值 |
| **输出 `tile_coord`** | TiledCopy 的归一化坐标：这个值在 Copy tile 中的位置 |

```
layout_TV:  (tid, vid) → mma_offset（原始 MMA 布局，可能不连续如 0,2,8,10...）
     ↓ composition(left_inverse(tile2mma), ...)
layout_tv:  (tid, vid) → tile_coord（归一化坐标 0,1,2,3...）
```

## 4. 逐步解析

### Step 1: 截断 layoutC_TV

```cpp
auto layoutC_TV = mma.get_layoutC_TV();
auto copy_V     = Int<Copy_Atom<CArgs...>::NumValSrc>{};
auto layout_TV  = composition(layoutC_TV, make_layout(make_shape(size<0>(layoutC_TV), copy_V)));
```

将 MMA 的完整布局截断到 Copy Atom 需要的值数量。

**示例**：
```
layoutC_TV: (32, 4) -> offset  // 32线程，每线程4个值
copy_V = 2                      // Copy Atom 每次处理2个值
layout_TV: (32, 2) -> offset    // 截断后
```

### Step 2: 计算 tiler

```cpp
auto tiler = transform(make_seq<rank(mma_tiler)>{}, [&](auto i) {
  return filter(composition(make_layout(mma_tiler, replace<i>(mma_zeros, Int<1>{})), layout_TV));
});
```

**分解理解**：

1. `make_layout(mma_tiler, replace<0>(mma_zeros, Int<1>{}))`：创建投影布局 `(M,N) -> M`
2. `composition(proj_m, layout_TV)`：得到 `(tid, vid) -> M坐标`
3. `filter(...)`：压缩冗余维度，得到覆盖的 M 坐标范围

**关键点**：`tiler = (tiler_m, tiler_n)` 是 `tuple<Layout, Layout>`，不是单个 2D Layout！

**filter 的作用**：
- 按 stride 从小到大排序，去除 stride=0 的模式
- 把二维 `(tid, vid)` 压缩成一维连续索引
- 使得相邻索引对应相邻内存位置

### Step 3: 计算 tile2mma

```cpp
auto tile2mma = composition(make_layout(mma_tiler), tiler);
```

当 `composition` 的 rhs 是 tuple 时，对每个维度分别处理：

```cpp
tile2mma(idx) = mma_layout(tiler_m(idx), tiler_n(idx))
              = tiler_m(idx) + tiler_n(idx) * M
```

### Step 4: 计算 layout_tv

```cpp
auto layout_tv = composition(left_inverse(tile2mma), layout_TV);
```

```
layout_TV:          (tid, vid) → mma_offset（跳跃的）
left_inverse:       mma_offset → tile_coord（连续的）
layout_tv:          (tid, vid) → tile_coord
```

**效果**：把 MMA 的跳跃式布局映射为连续的 Copy 索引。

## 5. 为什么需要这个转换？

MMA 累加器布局是由硬件 MMA 指令决定的，通常是"跳跃"的：

```
MMA 布局:
  线程 0 拥有 offset {0, 8, 64, 72}  ← 不连续
  线程 1 拥有 offset {1, 9, 65, 73}  ← 不连续

Copy 需要:
  相邻线程访问相邻地址（如 stmatrix 指令要求）
```

`layout_tv` 提供了这个"重编号"：

```
layout_tv 重映射后:
  线程 0 的 tile_coord {0, 2, 4, 6}
  线程 1 的 tile_coord {1, 3, 5, 7}
  相邻线程负责相邻的 tile_coord
```

## 6. 实际使用流程

```cpp
// 1. 创建"桥接"用的 TiledCopy
TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, tiled_mma);
//        layout_tv 记录了 MMA -> tile_coord 的映射

// 2. 用它作为参考，创建实际执行的 TiledCopy
TiledCopy tiled_r2s = make_tiled_copy_S(Copy_Atom<R2S>{}, tiled_copy_C_atom);
//                    ^^^^新的Copy指令    ^^^^保留 Src 布局

// 3. 执行 copy 时
//    - retile_S 知道如何从 MMA 寄存器读取（用 layout_tv 的输入侧）
//    - partition_D 知道如何写到 SMEM（用 layout_tv 的输出侧 + tiler）
Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);
Tensor tRS_sD   = thread_r2s.partition_D(sD_epi);
copy(tiled_r2s, tRS_rAcc, tRS_sD);
```

## 7. 测试代码

完整测试代码见 `cute-examples/make_tiled_copy_C_atom.cu`：

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <iostream>
#include <iomanip>

using namespace cute;

template <int copy_V, class LayoutC_TV, class MmaTiler>
void test_copy_v(const char* title, LayoutC_TV const& layoutC_TV, MmaTiler const& mma_tiler) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Step 1: Truncate layoutC_TV
    auto truncate_shape = make_shape(size<0>(layoutC_TV), Int<copy_V>{});
    auto truncate_layout = make_layout(truncate_shape);
    auto layout_TV = composition(layoutC_TV, truncate_layout);

    std::cout << "layout_TV = "; print(layout_TV); std::cout << "\n";
    std::cout << "含义: (thr, val) -> mma_offset\n\n";

    // Step 2: Compute tiler
    auto mma_zeros = repeat_like(mma_tiler, Int<0>{});

    // tiler_m
    auto stride_m = replace<0>(mma_zeros, Int<1>{});
    auto proj_m_layout = make_layout(mma_tiler, stride_m);
    auto tiler_m = filter(composition(proj_m_layout, layout_TV));

    // tiler_n
    auto stride_n = replace<1>(mma_zeros, Int<1>{});
    auto proj_n_layout = make_layout(mma_tiler, stride_n);
    auto tiler_n = filter(composition(proj_n_layout, layout_TV));

    auto tiler = make_tuple(tiler_m, tiler_n);
    std::cout << "tiler = ("; print(tiler_m); std::cout << ", "; print(tiler_n); std::cout << ")\n";
    std::cout << "tiler 是 tuple<Layout, Layout>，不是单个 2D Layout\n\n";

    // Step 3: Compute tile2mma
    auto mma_layout = make_layout(mma_tiler);
    auto tile2mma = composition(mma_layout, tiler);
    std::cout << "tile2mma = "; print(tile2mma); std::cout << "\n\n";

    // Step 4: Compute layout_tv
    auto inv_tile2mma = left_inverse(tile2mma);
    auto layout_tv = composition(inv_tile2mma, layout_TV);

    std::cout << "layout_tv = "; print(layout_tv); std::cout << "\n";
    std::cout << "含义: (thr, val) -> 归一化的 tile_coord\n\n";

    print_layout(layout_tv);
}

int main() {
    using MMA_Op = SM80_16x8x16_F16F16F16F16_TN;
    using MMA_Traits = MMA_Traits<MMA_Op>;
    using MMA_Atom = MMA_Atom<MMA_Traits>;

    auto tiled_mma = make_tiled_mma(MMA_Atom{},
                                     Layout<Shape<_1, _1, _1>>{},
                                     Tile<_16, _8, _16>{});

    auto layoutC_TV = tiled_mma.get_layoutC_TV();
    auto mma_tiler = make_shape(tile_size<0>(tiled_mma), tile_size<1>(tiled_mma));

    std::cout << "layoutC_TV: "; print(layoutC_TV); std::cout << "\n";
    std::cout << "mma_tiler: "; print(mma_tiler); std::cout << "\n";

    test_copy_v<1>("copy_V = 1", layoutC_TV, mma_tiler);
    test_copy_v<2>("copy_V = 2", layoutC_TV, mma_tiler);
    test_copy_v<4>("copy_V = 4", layoutC_TV, mma_tiler);

    return 0;
}
```

### 编译运行

```bash
nvcc -std=c++17 -I/path/to/cutlass/include \
     make_tiled_copy_C_atom.cu -o make_tiled_copy_C_atom
./make_tiled_copy_C_atom
```

## 8. 总结

```
┌─────────────────────────────────────────────────────────────────┐
│                  make_tiled_copy_C_atom                         │
├─────────────────────────────────────────────────────────────────┤
│  目的: 创建一个 TiledCopy，其布局匹配 MMA 累加器                  │
├─────────────────────────────────────────────────────────────────┤
│  输入:                                                           │
│    - copy_atom: Copy 指令封装                                    │
│    - mma: TiledMMA，提供累加器布局                                │
├─────────────────────────────────────────────────────────────────┤
│  输出: TiledCopy，包含:                                          │
│    - layout_tv: (tid, vid)_MMA -> tile_coord_Copy                │
│    - tiler: (tiler_m, tiler_n) 覆盖范围                          │
├─────────────────────────────────────────────────────────────────┤
│  核心步骤:                                                       │
│    1. 截断 layoutC_TV 到 copy_V                                  │
│    2. filter 提取 M/N 方向覆盖，组成 tiler                        │
│    3. composition(mma_layout, tiler) 得到 tile2mma               │
│    4. composition(left_inverse(tile2mma), layout_TV) 得到 layout_tv│
├─────────────────────────────────────────────────────────────────┤
│  关键理解:                                                       │
│    - layout_tv 输入是 MMA 的 (tid, vid)                          │
│    - layout_tv 输出是 Copy 的 tile_coord                         │
│    - 这是 MMA 布局到 Copy 布局的"翻译表"                          │
│    - filter 按 stride 排序，让相邻线程访问相邻位置                 │
└─────────────────────────────────────────────────────────────────┘
```

## References

- `include/cute/atom/copy_atom.hpp` - make_tiled_copy_C_atom 定义
- `include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` - 实际使用示例
- [0x14 TiledCopy tile2thrfrg](0x14-tiledcopy-tile2thrfrg.md) - 相关的 tile2thrfrg 函数
- [0x16 TiledCopy retile](0x16-tiledcopy-retile.md) - retile 函数详解
