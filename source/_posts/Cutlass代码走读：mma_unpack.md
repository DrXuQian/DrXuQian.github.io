---
title: CUTLASS 代码走读：mma_unpack 详解
date: 2024-12-24 11:00:00
tags:
  - CUTLASS
  - CuTe
  - GMMA
  - SM90
categories:
  - GPU Computing
---

本文解析 SM90 GMMA 特化的 `mma_unpack` 函数，理解如何将 Tensor 展开为底层 WGMMA 指令的寄存器参数。

<!-- more -->

## 1. 函数签名

```cpp
template <class MMA_Op, class... MMA_Args,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(MMA_Traits<MMA_Op, MMA_Args...> const& traits,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
```

### Step 1: 静态检查 - 必须是寄存器

```cpp
static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");
```

- `is_rmem` = is register memory
- 确保所有 tensor 都在寄存器中（不是 GMEM/SMEM 指针）

### Step 2: 提取寄存器类型

```cpp
using RegTypeA = typename remove_extent<typename MMA_Op::ARegisters>::type;
using RegTypeB = typename remove_extent<typename MMA_Op::BRegisters>::type;
using RegTypeC = typename remove_extent<typename MMA_Op::CRegisters>::type;
```

MMA_Op 定义了寄存器数组类型：

| 寄存器     | 定义          | remove_extent 后 |
| ---------- | ------------- | ---------------- |
| ARegisters | `uint32_t[4]` | `uint32_t`       |
| BRegisters | `uint64_t[1]` | `uint64_t`       |
| CRegisters | `uint32_t[8]` | `uint32_t`       |

### Step 3: GMMA 特殊性 - C 和 D 必须相同

```cpp
static_assert(is_same<typename TD::value_type, typename TC::value_type>::value, 
              "GMMA C and D value_type must match.");
static_assert(is_same<DLayout, CLayout>::value, 
              "GMMA C and D layouts must match.");
```

**关键区别：**

- 普通 MMA 四操作数：`D = C + A × B`
- SM90 GMMA 三操作数：`D = D + A × B`（原地累加）

因此 C 和 D 必须是同一个 tensor。

### Step 4: Recast tensor 到寄存器类型

```cpp
Tensor rA = recast<RegTypeA>(A);  // 重新解释 A 为 uint32_t 数组
Tensor rB = recast<RegTypeB>(B);  // 重新解释 B 为 uint64_t 数组
Tensor rC = recast<RegTypeC>(D);  // 用 D (可变) 而非 C
```

> **注意**：用 D 而非 C，因为 C 是 const，而 GMMA 需要原地修改。

### Step 5: 获取寄存器数量并检查

```cpp
constexpr int RegNumA = extent<typename MMA_Op::ARegisters>::value;  // 4
constexpr int RegNumB = extent<typename MMA_Op::BRegisters>::value;  // 1
constexpr int RegNumC = extent<typename MMA_Op::CRegisters>::value;  // 8

CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});
CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});
```

| Tensor | extent | 含义                   |
| ------ | ------ | ---------------------- |
| rA     | 4      | 4 个 32-bit 寄存器     |
| rB     | 1      | 1 个 64-bit descriptor |
| rC     | 8      | 8 个 32-bit 寄存器     |

### Step 6: explode 展开并调用 fma

```cpp
detail::explode(MMA_Op::fma,
                rA, make_int_sequence<RegNumA>{},
                rB, make_int_sequence<RegNumB>{},
                rC, make_int_sequence<RegNumC>{},
                &(traits.accumulate_), seq<0>{});
```

`explode` 将 tensor 元素展开为函数参数：

```cpp
// 输入
explode(fma, rA, <0,1,2,3>, rB, <0>, rC, <0,1,2,3,4,5,6,7>, &scale, <0>)

// 展开为
fma(rA[0], rA[1], rA[2], rA[3],           // A 的 4 个寄存器
    rB[0],                                 // B 的 1 个 descriptor
    rC[0], rC[1], rC[2], rC[3],           // C 的 8 个寄存器
    rC[4], rC[5], rC[6], rC[7],
    scale)                                 // scale 参数
```

### 总结

`mma_unpack` 的核心工作：

1. **类型检查**：确保数据在寄存器中
2. **类型转换**：recast 成底层寄存器类型
3. **展开调用**：`explode` 将 tensor 元素展开为 `fma()` 参数
4. **执行指令**：调用 PTX WGMMA 指令

> **注意**：GMMA 是三操作数指令（D = D + A × B），所以 C 和 D 必须是同一个 tensor。