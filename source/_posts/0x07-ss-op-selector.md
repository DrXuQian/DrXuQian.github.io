---
title: "0x07 CUTLASS ss_op_selector MMA Atom Selection"
date: 2024-12-24 07:00:00
tags:
  - CUTLASS
  - CuTe
  - GMMA
  - SM90
categories:
  - GPU Computing
---

This article explains `ss_op_selector` function, understanding how CUTLASS selects appropriate GMMA Atom based on data types and TileShape.

<!-- more -->

## 1. 简介

这个函数用于选择 Atom MMA operation，调用在 MMA的CollectiveBuilder。

一些用这个function进行Atom MMA的选择的例子:

| ElementA | ElementB | TileShape     | GMMA Atom         |
| :------- | -------- | ------------- | ----------------- |
| half     | half     | (128,128,64)  | SM90_64x128x16_SS |
| half     | half     | (128,256,64)  | SM90_64x256x16_SS |
| bf16     | bf16     | (128,128,64)  | SM90_64x128x16_SS |
| tfloat32 | tfloat32 | (128,128,32)  | SM90_64x128x8_SS  |
| fp8_e4m3 | fp8_e4m3 | (128,128,128) | SM90_64x128x32_SS |

## 定义

首先，看一下这个函数的定义（include/cute/arch/mma_sm90.hpp）：

```
	  template <
	  class ElementA,
	  class ElementB,
	  class ElementC,
	  class TileShape_MNK,
	  GMMA::Major MajorA = GMMA::Major::K,
	  GMMA::Major MajorB = GMMA::Major::K,
	  auto... Args                         // e.g. GMMA::ScaleOut::One, [GMMA::ScaleIn::One, GMMA::ScaleIn::One]
	                                       // But most commonly leave empty for defaults
	  >
	  CUTE_HOST_DEVICE constexpr
	  auto
	  ss_op_selector()
```

可以看到这个函数没有接受参数。template的parameter list中，有一个`auto... Args`，这个是什么意思：

`auto... Args` 是 C++17 的可变参数 auto 语法，这个代表的是可变长的参数列表，但是这里的列表本身的内容还是编译期常量，只是可以处理不同长度的编译期常量列表。

一个简单的例子：

```
template <auto... Values>
struct ValueList {};

ValueList<1, 2, 3>           // 三个 int
ValueList<1, 'a', true>      // 混合: int, char, bool
ValueList<>                  // 空
```

其他的参数就是一些对于Atom MMA的类型进行选择的选项。包括A/B/C的数据格式，TileShape_MNK也就是CollectiveMma中单个stage计算的矩阵大小。MajorA和MajorB是A和B的cutlass语境下的layout格式。

## 代码走读

1. 首先是一个assertion，对于gmma，M必须是64的整数倍。

```
  static_assert(size<0>(TileShape_MNK{}) % 64 == 0, "Tile_M must be a multiple of 64.");
```

2. 后续就是大量的conditon branch，来选择不同的Atom MMA的函数：

```
auto Tile_N = size<1>(TileShape_MNK{});

  // F16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
...
```

可以看到这里同样有`Args...`，这里是把Arg参数解包传入下面的template进行实例化。

