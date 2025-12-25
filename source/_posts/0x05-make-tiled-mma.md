---
title: "0x05 CUTLASS make_tiled_mma and MMA_Atom"
date: 2024-12-24 05:00:00
tags:
  - CUTLASS
  - CuTe
  - TiledMMA
  - SM90
categories:
  - GPU Computing
---

This article explains `make_tiled_mma` function and `MMA_Atom` class implementation, understanding how to build TiledMMA from Atom MMA.

<!-- more -->

> **示例代码**: [0x05_make_tiled_mma.cu](https://github.com/DrXuQian/cute-examples/blob/main/0x05_make_tiled_mma.cu)

## 1. make_tiled_mma 函数

```C++
//
// These tile the MMA_Atom as a whole
//

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Atom<MMA_Op> const& mma_atom,
               MMAThrLayout     const& thr_layout   = {},
               Permutations     const& permutations = {})
{
  // append<3>: 确保 layout 有 3 个维度
  // 如果不足 3 维，用 Layout<_1,_0>{} 填充

  // ═══════════════════════════════════════════════════════════════════════════
  //                    Layout<_1, _0> 含义
  // ═══════════════════════════════════════════════════════════════════════════
  auto thr_layout_mnk  = append<3>(thr_layout, Layout<_1,_0>{});
  auto permutation_mnk = append<3>(permutations, _);

  return TiledMMA<MMA_Atom<MMA_Op>,
                  decltype(thr_layout_mnk),
                  decltype(permutation_mnk)>{mma_atom, thr_layout_mnk};
}

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Op       const&,
               MMAThrLayout const& thr_layout   = {},
               Permutations const& permutations = {})
{
  // Attempt to wrap in an MMA_Atom<> and forward
  return make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations);
}

```

代码入口就是上面两种，第二种中间调用了第一种，只是为了额外支持单独的MMA_Op的输入，在内部裹成了MMA_Atom类型。输入进`make_tiled_mma`。

可以看到第一个代码内部需要将thr_layout转换成三维的layout。permutation也是一样。这里`append<3>(thr_layout, Layout<_1,_0>{});`后面的layout的1是shape，0是stride。

## MMA_Atom class

### 类的定义

class的定义如下：

```
template <class... Args>
struct MMA_Atom;

template <class MMAOperation>
struct MMA_Atom<MMAOperation> : MMA_Atom<MMA_Traits<MMAOperation>>
{};

template <class MMAOperation, class... Args>
struct MMA_Atom<MMA_Traits<MMAOperation, Args...>>
  : MMA_Traits<MMAOperation, Args...>
{
```

这里的类非常复杂，看最后这里的类定义，首先是一个full specialization，输入的类型是`MMA_Traits<MMAOperation, Args...>`, 这里对于MMA_Traits类用MMA_Operation 进行了实例化。然后还继承了这个traits类。这是一个常用的用template的参数class作为基类的例子。

这样，MMA_Atom就继承了Traits类，这样可以拿到Traits内部的很多MMA Atom的信息。

### Type Alias

然后是一堆Type Alias，把Traits类内部的参数用using的方式取了别名。

```C++
  // Element value types from the MMA_Traits
  using ValTypeD = typename Traits::ValTypeD;
  using ValTypeA = typename Traits::ValTypeA;
  using ValTypeB = typename Traits::ValTypeB;
  using ValTypeC = typename Traits::ValTypeC;

  // Thr-Val layouts from the MMA_Traits
  using Shape_MNK  = typename Traits::Shape_MNK;
  using ThrID      = typename Traits::ThrID;
  using LayoutC_TV = typename Traits::CLayout;
  using LayoutA_TV = typename Traits::ALayout;
  using LayoutB_TV = typename Traits::BLayout;

  // Fragment value types from the MMA_Traits (optional, defaults to Val type)
  using FrgTypeD = typename detail::FrgTypeC_or_Default<Traits>::type;
  using FrgTypeA = typename detail::FrgTypeA_or_Default<Traits>::type;
  using FrgTypeB = typename detail::FrgTypeB_or_Default<Traits>::type;
  using FrgTypeC = typename detail::FrgTypeC_or_Default<Traits>::type;
```

### With函数

然后定义了一个with函数，用来对于MMA_Atom中的某些参数进行修改之后，返回一个新的类，这样就不需要重复构建atom类。其他的变种都可以用with接口进行重用。

```C++
  // Additional Trait parameters/transformations
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE
  auto
  with(TraitsArgs&&... args) const {
    auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
    return MMA_Atom<decltype(traits)>{traits};
  }
```

### Call函数

然后是定义了AtomMMA的调用函数:

```C++
  // Cast, check, and call fma
  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout> const& C) const
  {
    static_assert(DLayout::rank == 1, "Expected rank-1 D tensor");
    static_assert(ALayout::rank == 1, "Expected rank-1 A tensor");
    static_assert(BLayout::rank == 1, "Expected rank-1 B tensor");
    static_assert(CLayout::rank == 1, "Expected rank-1 C tensor");

    return mma_unpack(static_cast<Traits const&>(*this), D, A, B, C);
  }

  // Three arguments reproduces C
  template <class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout>      & C) const
  {
    return call(C, A, B, C);
  }
```

注意上面的两种variation。

### 构建fragment A/B/C

然后是对于fragmentA和fragmentB以及fragmentC的构建函数。

