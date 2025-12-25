---
title: "0x03 ABLayout vs ALayout_64x16 - SS and RS Mode TV Layout"
date: 2024-12-24 03:00:00
categories: [CUDA, CuTe, CUTLASS]
tags: [layout, wgmma, ss-mode, rs-mode, tv-layout]
---

This article compares two TV Layout types for WGMMA operand access: ABLayout (SS mode) vs ALayout_64x16 (RS mode).

<!-- more -->

> **示例代码**: [0x03_ablayout_vs_alayout.cu](https://github.com/DrXuQian/cute-examples/blob/main/0x03_ablayout_vs_alayout.cu)

## 1. Introduction

In CUTLASS WGMMA (Warpgroup Matrix Multiply-Accumulate), there are two operand access modes:
- **SS mode**: A and B both in Shared Memory, uses `ABLayout`
- **RS mode**: A in Register, B in Shared Memory, uses `ALayout_64x16`

This article compares these two TV Layout structures, mapping functions, and semantic differences.

## 2. Layout 定义

### 2.1 ABLayout (SS 模式)

```cpp
// Shared memory source layouts for any value type
template <int M, int K>
using ABLayout = Layout<Shape <_128, Shape <Int<M>, Int<K>>>,
                        Stride<  _0, Stride<    _1, Int<M>>>>;
```

### 2.2 ALayout_64x16 (RS 模式)

```cpp
// Register source layout for 16-bit (sparse 32-bit) value types
using ALayout_64x16 = Layout<Shape <Shape <  _4, _8, _4>, Shape < _2, _2,  _2>>,
                             Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;
```

---

## 3. ABLayout 详解

### 3.1 实例化 ABLayout<64, 16>

```cpp
ABLayout<64, 16> = Layout<
    Shape <_128,        Shape <_64, _16>>,    // (Thr, (Val_M, Val_K))
    Stride<  _0,        Stride< _1, _64>>     // (Thr_stride, (Val_M_stride, Val_K_stride))
>;
```

### 3.2 维度分解

| 维度 | Shape | Stride | 含义 |
|------|-------|--------|------|
| **Thr** | 128 | **0** | 128 个线程，stride=0 |
| **Val_M** | 64 | 1 | M 方向 64 行 |
| **Val_K** | 16 | 64 | K 方向 16 列 |

### 3.3 映射函数

TV Layout 本质是一个函数：

$$
f: (\text{thread\_id}, \text{value\_id}) \rightarrow \text{offset}
$$

对于 ABLayout：

$$
\text{offset}(\text{thr}, v_m, v_k) = \text{thr} \times 0 + v_m \times 1 + v_k \times 64
$$

**简化为：**

$$
\boxed{\text{offset}(v_m, v_k) = v_m + v_k \times 64}
$$

### 3.4 Thr Stride = 0 的含义

这是 ABLayout 最关键的特性：

```
Thread Stride = 0 意味着:

  offset(thread_0,   v_m, v_k) = v_m + v_k × 64
  offset(thread_1,   v_m, v_k) = v_m + v_k * 64   <- same!
  offset(thread_2,   v_m, v_k) = v_m + v_k * 64   <- same!
  ...
  offset(thread_127, v_m, v_k) = v_m + v_k * 64   <- same!

+-------------------------------------------------------------+
|       All 128 threads compute the same offset!              |
|                                                             |
|       They share the same SMEM data block                   |
+-------------------------------------------------------------+
```

### 3.5 64x16 Matrix Offset Distribution

```
64x16 Matrix (Column-Major, shared by all threads):

        k=0    k=1    k=2    ...    k=15
       +------+------+------+------+------+
  m=0  |   0  |  64  | 128  | ...  | 960  |
       +------+------+------+------+------+
  m=1  |   1  |  65  | 129  | ...  | 961  |
       +------+------+------+------+------+
  m=2  |   2  |  66  | 130  | ...  | 962  |
       +------+------+------+------+------+
  ...  | ...  | ...  | ...  | ...  | ...  |
       +------+------+------+------+------+
  m=63 |  63  | 127  | 191  | ...  | 1023 |
       +------+------+------+------+------+

       offset = m + k x 64

       Total 64 x 16 = 1024 elements
       All 128 threads can access the entire matrix
```

### 3.6 SS Mode Data Flow

```
+-------------------------------------------------------------+
|                    SS Mode (ABLayout)                       |
+-------------------------------------------------------------+
|                                                             |
|  GMEM --TMA--> SMEM --descriptor--> WGMMA Hardware          |
|                  ^                                          |
|           64-bit descriptor                                 |
|           points to SMEM start address                      |
|                  ^                                          |
|         shared by all 128 threads                           |
|                                                             |
|  Feature: data stays in SMEM, HW reads directly, no load    |
|                                                             |
+-------------------------------------------------------------+
```

---

## 4. ALayout_64x16 详解

### 4.1 结构分解

```cpp
ALayout_64x16 = Layout<
    Shape <Shape <  _4, _8, _4>, Shape < _2, _2,  _2>>,
    Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>
>;
```

| 部分 | Shape | Stride | 元素数 |
|------|-------|--------|--------|
| **Thr** | (4, 8, 4) | (128, 1, 16) | 4×8×4 = 128 |
| **Val** | (2, 2, 2) | (64, 8, 512) | 2×2×2 = 8 |

**总元素数**: 128 线程 × 8 值/线程 = 1024 = 64 × 16 ✓

### 4.2 坐标展开公式

**Thread ID → (t₀, t₁, t₂):**

$$
t_0 = \text{thread\_id} \mod 4
$$
$$
t_1 = \lfloor \text{thread\_id} / 4 \rfloor \mod 8
$$
$$
t_2 = \lfloor \text{thread\_id} / 32 \rfloor \mod 4
$$

**Value ID → (v₀, v₁, v₂):**

$$
v_0 = \text{value\_id} \mod 2
$$
$$
v_1 = \lfloor \text{value\_id} / 2 \rfloor \mod 2
$$
$$
v_2 = \lfloor \text{value\_id} / 4 \rfloor \mod 2
$$

### 4.3 映射函数

$$
\text{offset} = \underbrace{t_0 \times 128 + t_1 \times 1 + t_2 \times 16}_{\text{thr\_offset}} + \underbrace{v_0 \times 64 + v_1 \times 8 + v_2 \times 512}_{\text{val\_offset}}
$$

### 4.4 Thread 0 的 8 个 Value

Thread 0: $t_0=0, t_1=0, t_2=0$ → thr_offset = 0

| v_idx | (v₀, v₁, v₂) | val_offset | offset | m | k |
|-------|--------------|------------|--------|---|---|
| 0 | (0, 0, 0) | 0 | 0 | 0 | 0 |
| 1 | (1, 0, 0) | 64 | 64 | 0 | 1 |
| 2 | (0, 1, 0) | 8 | 8 | 8 | 0 |
| 3 | (1, 1, 0) | 72 | 72 | 8 | 1 |
| 4 | (0, 0, 1) | 512 | 512 | 0 | 8 |
| 5 | (1, 0, 1) | 576 | 576 | 0 | 9 |
| 6 | (0, 1, 1) | 520 | 520 | 8 | 8 |
| 7 | (1, 1, 1) | 584 | 584 | 8 | 9 |

*坐标转换: m = offset % 64, k = offset / 64*

### 4.5 Thread 0 在 64×16 矩阵中的位置

```
64x16 matrix (Thread 0's 8 positions marked with *):

           k=0       k=1                   k=8       k=9
          +---------+---------+           +---------+---------+
     m=0  | V0  *   | V1  *   |    ...    | V4  *   | V5  *   |
          | off=0   | off=64  |           | off=512 | off=576 |
          +---------+---------+           +---------+---------+
     m=1  |         |         |           |         |         |
          +---------+---------+           +---------+---------+
     ...  |         |         |           |         |         |
          +---------+---------+           +---------+---------+
     m=8  | V2  *   | V3  *   |    ...    | V6  *   | V7  *   |
          | off=8   | off=72  |           | off=520 | off=584 |
          +---------+---------+           +---------+---------+
     ...  |         |         |           |         |         |
          +---------+---------+           +---------+---------+

Thread 0 owns:
  - 4 positions at row m=0: (0,0), (0,1), (0,8), (0,9)
  - 4 positions at row m=8: (8,0), (8,1), (8,8), (8,9)
```

### 4.6 Thread 1 的 8 个 Value

Thread 1: $t_0=1, t_1=0, t_2=0$ → thr_offset = 1×128 = 128

| v_idx | val_offset | offset | m | k |
|-------|------------|--------|---|---|
| 0 | 0 | 128 | 0 | 2 |
| 1 | 64 | 192 | 0 | 3 |
| 2 | 8 | 136 | 8 | 2 |
| 3 | 72 | 200 | 8 | 3 |
| 4 | 512 | 640 | 0 | 10 |
| 5 | 576 | 704 | 0 | 11 |
| 6 | 520 | 648 | 8 | 10 |
| 7 | 584 | 712 | 8 | 11 |

### 4.7 完整线程分布图

```
64x16 matrix (showing which Thread owns each position):

        k: 0    1  | 2    3  | 4    5  | 6    7  || 8    9  |10   11  |12   13  |14   15
       +---------+---------+---------+---------++---------+---------+---------+---------+
  m=0  | T0   T0 | T1   T1 | T2   T2 | T3   T3 || T0   T0 | T1   T1 | T2   T2 | T3   T3 |
       +---------+---------+---------+---------++---------+---------+---------+---------+
  m=1  | T4   T4 | T5   T5 | T6   T6 | T7   T7 || T4   T4 | T5   T5 | T6   T6 | T7   T7 |
       +---------+---------+---------+---------++---------+---------+---------+---------+
  ...  |   ...   |   ...   |   ...   |   ...   ||   ...   |   ...   |   ...   |   ...   |
       +---------+---------+---------+---------++---------+---------+---------+---------+
  m=8  | T0   T0 | T1   T1 | T2   T2 | T3   T3 || T0   T0 | T1   T1 | T2   T2 | T3   T3 |
       +---------+---------+---------+---------++---------+---------+---------+---------+
  m=9  | T4   T4 | T5   T5 | T6   T6 | T7   T7 || T4   T4 | T5   T5 | T6   T6 | T7   T7 |
       +---------+---------+---------+---------++---------+---------+---------+---------+

Pattern:
  - k direction: T0-T3 alternate, each thread covers 2 columns
  - m direction: same thread repeats every 8 rows
  - each thread's 8 values spread across 4 2x2 blocks
```

### 4.8 RS Mode Data Flow

```
+-------------------------------------------------------------+
|                    RS Mode (ALayout_64x16)                  |
+-------------------------------------------------------------+
|                                                             |
|  GMEM --TMA--> SMEM --ldmatrix--> Register --> WGMMA HW     |
|                          ^            ^                     |
|                    each thread    each thread 8 values      |
|                    loads its 8    stored in register        |
|                          ^                                  |
|                  ALayout_64x16 specifies                    |
|                  which positions each thread owns           |
|                                                             |
|  Feature: data needs to be loaded from SMEM to Register     |
|                                                             |
+-------------------------------------------------------------+
```

---

## 5. 核心对比

### 5.1 公式对比

| Layout | 映射函数 |
|--------|----------|
| **ABLayout<64,16>** | $\text{offset} = v_m + v_k \times 64$ |
| **ALayout_64x16** | $\text{offset} = (t_0 \times 128 + t_1 + t_2 \times 16) + (v_0 \times 64 + v_1 \times 8 + v_2 \times 512)$ |

### 5.2 Thr Stride 对比

| Layout | Thr Stride | 含义 |
|--------|------------|------|
| **ABLayout** | **0** | 所有线程共享相同数据 |
| **ALayout_64x16** | (128, 1, 16) ≠ 0 | 每线程负责不同位置 |

### 5.3 数据位置对比

```
+-------------------------------------------------------------+
|                    ABLayout (SS)                            |
+-------------------------------------------------------------+
|                                                             |
|  Thread 0   --> SMEM descriptor --> access entire 64x16     |
|  Thread 1   --> SMEM descriptor --> access entire 64x16     |
|  Thread 2   --> SMEM descriptor --> access entire 64x16     |
|  ...                                                        |
|  Thread 127 --> SMEM descriptor --> access entire 64x16     |
|                                                             |
|  Data location: SMEM (no movement)                          |
|  Engine: smem_desc<T> (64-bit descriptor)                   |
|                                                             |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                  ALayout_64x16 (RS)                         |
+-------------------------------------------------------------+
|                                                             |
|  Thread 0   --> Register[0..7] --> pos (0,0)(0,1)(8,0)...   |
|  Thread 1   --> Register[0..7] --> pos (0,2)(0,3)(8,2)...   |
|  Thread 2   --> Register[0..7] --> pos (0,4)(0,5)(8,4)...   |
|  ...                                                        |
|  Thread 127 --> Register[0..7] --> different 8 positions    |
|                                                             |
|  Data location: Register (needs SMEM load)                  |
|  Engine: ArrayEngine<uint32_t, 4>                           |
|                                                             |
+-------------------------------------------------------------+
```

### 5.4 Summary Diagram

```
                    ABLayout (SS)                    ALayout_64x16 (RS)
                   +-------------+                   +-------------+
                   |             |                   |             |
 Thr Stride        |      0      |                   |    != 0     |
                   |             |                   |             |
                   +------+------+                   +------+------+
                          |                                 |
                          v                                 v
                 +----------------+               +----------------+
 Thread View     | all threads    |               | each thread    |
                 | see same off   |               | sees diff off  |
                 +-------+--------+               +-------+--------+
                          |                                 |
                          v                                 v
                 +----------------+               +----------------+
 Data Location   |     SMEM       |               |    Register    |
                 |  (descriptor)  |               | (per-thread)   |
                 +-------+--------+               +-------+--------+
                          |                                 |
                          v                                 v
                 +----------------+               +----------------+
 Engine          |  smem_desc<T>  |               | ArrayEngine    |
                 |   (64-bit)     |               |  <T, 8>        |
                 +----------------+               +----------------+
```

---

## 6. 为什么需要两种模式？

### 6.1 SS 模式的优势

- **简单**: 数据不移动，硬件直接读 SMEM
- **省寄存器**: 不需要额外寄存器存储 A 矩阵
- **适合大矩阵**: SMEM 容量大

### 6.2 RS 模式的优势

- **复用寄存器数据**: A 矩阵在 register 中可多次使用
- **减少 SMEM 访问**: 一次 load，多次 MMA
- **适合 K 维度循环**: 同一份 A 数据参与多个 K 步的计算

### 6.3 选择建议

| 场景 | 推荐模式 |
|------|----------|
| A 只用一次 | SS |
| A 需要复用 | RS |
| 寄存器紧张 | SS |
| SMEM 带宽瓶颈 | RS |

---

## 7. 总结

| 特性 | ABLayout (SS) | ALayout_64x16 (RS) |
|------|---------------|---------------------|
| **Thr Stride** | 0 | ≠ 0 |
| **线程数据** | 所有线程共享 | 每线程独立 8 个值 |
| **数据位置** | SMEM | Register |
| **Engine 类型** | smem_desc | ArrayEngine |
| **数据移动** | 无 | SMEM → Register |
| **映射函数** | offset = m + k×M | 复杂多维映射 |

**核心理解：**
- `ABLayout` 的 **Thr Stride = 0** 意味着所有线程共享同一块 SMEM，是 WGMMA SS 模式的布局描述
- `ALayout_64x16` 的 **Thr Stride ≠ 0** 意味着每个线程负责矩阵的不同位置，需要从 SMEM load 到 Register，是 WGMMA RS 模式的布局描述
