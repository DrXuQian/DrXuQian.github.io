---
title: "0x04 CUTLASS Atom MMA (SS/RS Mode)"
date: 2024-12-24 10:00:00
tags:
  - CUTLASS
  - CuTe
  - GMMA
  - SM90
categories:
  - GPU Computing
---

This article explains CUTLASS Atom MMA SS and RS modes, including WGMMA instruction parameters and MMA_Traits definitions.

<!-- more -->

## 1. MMA SS 模式

Atom MMA的一个例子：

```c++
// GMMA 64x256x16 F16+=F16*F16
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
struct MMA_64x256x16_F16F16F16_SS
{
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t      & d00, uint32_t      & d01, uint32_t      & d02, uint32_t      & d03,
      uint32_t      & d04, uint32_t      & d05, uint32_t      & d06, uint32_t      & d07,
      uint32_t      & d08, uint32_t      & d09, uint32_t      & d10, uint32_t      & d11,
      uint32_t      & d12, uint32_t      & d13, uint32_t      & d14, uint32_t      & d15,
      uint32_t      & d16, uint32_t      & d17, uint32_t      & d18, uint32_t      & d19,
      uint32_t      & d20, uint32_t      & d21, uint32_t      & d22, uint32_t      & d23,
      uint32_t      & d24, uint32_t      & d25, uint32_t      & d26, uint32_t      & d27,
      uint32_t      & d28, uint32_t      & d29, uint32_t      & d30, uint32_t      & d31,
      uint32_t      & d32, uint32_t      & d33, uint32_t      & d34, uint32_t      & d35,
      uint32_t      & d36, uint32_t      & d37, uint32_t      & d38, uint32_t      & d39,
      uint32_t      & d40, uint32_t      & d41, uint32_t      & d42, uint32_t      & d43,
      uint32_t      & d44, uint32_t      & d45, uint32_t      & d46, uint32_t      & d47,
      uint32_t      & d48, uint32_t      & d49, uint32_t      & d50, uint32_t      & d51,
      uint32_t      & d52, uint32_t      & d53, uint32_t      & d54, uint32_t      & d55,
      uint32_t      & d56, uint32_t      & d57, uint32_t      & d58, uint32_t      & d59,
      uint32_t      & d60, uint32_t      & d61, uint32_t      & d62, uint32_t      & d63,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    cutlass::arch::synclog_emit_wgmma_smem_smem(__LINE__, desc_a, desc_b);
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
      " %64,"
      " %65,"
      " p,    %67,  %68,  %69,  %70;\n"
    "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03),
        "+r"(d04), "+r"(d05), "+r"(d06), "+r"(d07),
        "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15),
        "+r"(d16), "+r"(d17), "+r"(d18), "+r"(d19),
        "+r"(d20), "+r"(d21), "+r"(d22), "+r"(d23),
        "+r"(d24), "+r"(d25), "+r"(d26), "+r"(d27),
        "+r"(d28), "+r"(d29), "+r"(d30), "+r"(d31),
        "+r"(d32), "+r"(d33), "+r"(d34), "+r"(d35),
        "+r"(d36), "+r"(d37), "+r"(d38), "+r"(d39),
        "+r"(d40), "+r"(d41), "+r"(d42), "+r"(d43),
        "+r"(d44), "+r"(d45), "+r"(d46), "+r"(d47),
        "+r"(d48), "+r"(d49), "+r"(d50), "+r"(d51),
        "+r"(d52), "+r"(d53), "+r"(d54), "+r"(d55),
        "+r"(d56), "+r"(d57), "+r"(d58), "+r"(d59),
        "+r"(d60), "+r"(d61), "+r"(d62), "+r"(d63)
      :  "l"(desc_a),
         "l"(desc_b),
         "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use MMA_64x256x16_F16F16F16_SS without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
  }
};


```

可以看到这里存在输入是scaleA和scaleB，类型是GMMA::ScaleIn，这里的取值可以是1或者-1。

```
enum class ScaleIn {
  Neg = -1,
  One =  1
};
```

另外可以看到fma函数中的输入参数为:

1. desc_a: TMA descriptor
2. desc_b: TMA descriptor
3. d矩阵的0-63的register
4. d矩阵的ScaleOut

其中ScaleOut取值为：

```
enum class ScaleOut {
  Zero = 0,
  One  = 1
};
```

可以简单认为，第一次MMA选Zero，后续都要配置为One。

## MMA RS

```C++
// GMMA 64x256x16 F16+=F16*F16
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
struct MMA_64x256x16_F16F16F16_RS
{
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  static_assert(tnspA == GMMA::Major::K,
      "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& a00, uint32_t const& a01, uint32_t const& a02, uint32_t const& a03,
      uint64_t const& desc_b,
      uint32_t      & d00, uint32_t      & d01, uint32_t      & d02, uint32_t      & d03,
      uint32_t      & d04, uint32_t      & d05, uint32_t      & d06, uint32_t      & d07,
      uint32_t      & d08, uint32_t      & d09, uint32_t      & d10, uint32_t      & d11,
      uint32_t      & d12, uint32_t      & d13, uint32_t      & d14, uint32_t      & d15,
      uint32_t      & d16, uint32_t      & d17, uint32_t      & d18, uint32_t      & d19,
      uint32_t      & d20, uint32_t      & d21, uint32_t      & d22, uint32_t      & d23,
      uint32_t      & d24, uint32_t      & d25, uint32_t      & d26, uint32_t      & d27,
      uint32_t      & d28, uint32_t      & d29, uint32_t      & d30, uint32_t      & d31,
      uint32_t      & d32, uint32_t      & d33, uint32_t      & d34, uint32_t      & d35,
      uint32_t      & d36, uint32_t      & d37, uint32_t      & d38, uint32_t      & d39,
      uint32_t      & d40, uint32_t      & d41, uint32_t      & d42, uint32_t      & d43,
      uint32_t      & d44, uint32_t      & d45, uint32_t      & d46, uint32_t      & d47,
      uint32_t      & d48, uint32_t      & d49, uint32_t      & d50, uint32_t      & d51,
      uint32_t      & d52, uint32_t      & d53, uint32_t      & d54, uint32_t      & d55,
      uint32_t      & d56, uint32_t      & d57, uint32_t      & d58, uint32_t      & d59,
      uint32_t      & d60, uint32_t      & d61, uint32_t      & d62, uint32_t      & d63,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    cutlass::arch::synclog_emit_wgmma_reg_smem(__LINE__, desc_b);
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
      "{%64,  %65,  %66,  %67},"
      " %68,"
      " p,    %70,  %71,  %72;\n"
    "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03),
        "+r"(d04), "+r"(d05), "+r"(d06), "+r"(d07),
        "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15),
        "+r"(d16), "+r"(d17), "+r"(d18), "+r"(d19),
        "+r"(d20), "+r"(d21), "+r"(d22), "+r"(d23),
        "+r"(d24), "+r"(d25), "+r"(d26), "+r"(d27),
        "+r"(d28), "+r"(d29), "+r"(d30), "+r"(d31),
        "+r"(d32), "+r"(d33), "+r"(d34), "+r"(d35),
        "+r"(d36), "+r"(d37), "+r"(d38), "+r"(d39),
        "+r"(d40), "+r"(d41), "+r"(d42), "+r"(d43),
        "+r"(d44), "+r"(d45), "+r"(d46), "+r"(d47),
        "+r"(d48), "+r"(d49), "+r"(d50), "+r"(d51),
        "+r"(d52), "+r"(d53), "+r"(d54), "+r"(d55),
        "+r"(d56), "+r"(d57), "+r"(d58), "+r"(d59),
        "+r"(d60), "+r"(d61), "+r"(d62), "+r"(d63)
      :  "r"(a00),  "r"(a01),  "r"(a02),  "r"(a03),
         "l"(desc_b),
         "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspB)));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use MMA_64x256x16_F16F16F16_RS without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
  }
};
```

这里多出了4个uint32的a的register，这是因为A矩阵是64x16，对于一个warpgroup，也就是128个thread来说，每个thread分配到了8个fp16的输入，也就是4个uint32的输入的register。输出是64x256，对于128个thread来说，每个thread是128个fp16，也就是64个uint32的d的register。

这两个MMA的Traits分别是：

```C++
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x16_F16F16F16_SS = SM90::GMMA::MMA_64x256x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x16_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};
```



```c++
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
using SM90_64x256x16_F16F16F16_RS = SM90::GMMA::MMA_64x256x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>;

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x256x16_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_256,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<256, 16>;
  using CLayout = GMMA::CLayout_64x256;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};
```

