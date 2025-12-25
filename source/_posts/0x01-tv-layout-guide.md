---
title: "0x01 TV Layout (Thread-Value Layout) Guide"
date: 2024-12-24 01:00:00
categories: [CUDA, CuTe, CUTLASS]
tags: [layout, thread, value, mma]
---

This article explains TV Layout in CuTe, the core abstraction for describing how data is distributed across threads.

<!-- more -->

> **示例代码**: [0x01_tv_layout.cu](https://github.com/DrXuQian/cute-examples/blob/main/0x01_tv_layout.cu)

## 1. Introduction

TV Layout is the core abstraction in CuTe that describes how data is distributed across threads. Understanding TV Layout is key to mastering WGMMA and TMA.

## 2. TV Layout 的本质

**TV Layout 本质上是一个函数：**

```
f: (thread_id, value_id) → offset

给定 (线程ID, 值索引) → 返回内存偏移
```

```cpp
// Layout 定义
Layout<Shape<Thr_shape, Val_shape>, Stride<Thr_stride, Val_stride>>

// 等价于函数:
offset = f(thread_id, value_id)
       = thread_id • Thr_stride + value_id • Val_stride
```

## 3. 简单示例

### 3.1 连续分布

```cpp
// 32 个元素分配给 4 个线程，每线程 8 个值
using TVLayout = Layout<
    Shape <_4, _8>,      // (Thr=4, Val=8)
    Stride<_8, _1>       // (Thr_stride=8, Val_stride=1)
>;

// offset = thread_idx * 8 + value_idx * 1
```

**TV Table:**

```
           Val: 0   1   2   3   4   5   6   7
              +---+---+---+---+---+---+---+---+
    Thr 0     | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
              +---+---+---+---+---+---+---+---+
    Thr 1     | 8 | 9 |10 |11 |12 |13 |14 |15 |
              +---+---+---+---+---+---+---+---+
    Thr 2     |16 |17 |18 |19 |20 |21 |22 |23 |
              +---+---+---+---+---+---+---+---+
    Thr 3     |24 |25 |26 |27 |28 |29 |30 |31 |
              +---+---+---+---+---+---+---+---+
```

### 3.2 交错分布

```cpp
using TVLayout = Layout<
    Shape <_4, _8>,      // (Thr=4, Val=8)
    Stride<_1, _4>       // (Thr_stride=1, Val_stride=4)
>;

// offset = thread_idx * 1 + value_idx * 4
```

**TV Table:**

```
           Val: 0   1   2   3   4   5   6   7
              +---+---+---+---+---+---+---+---+
    Thr 0     | 0 | 4 | 8 |12 |16 |20 |24 |28 |
              +---+---+---+---+---+---+---+---+
    Thr 1     | 1 | 5 | 9 |13 |17 |21 |25 |29 |
              +---+---+---+---+---+---+---+---+
    Thr 2     | 2 | 6 |10 |14 |18 |22 |26 |30 |
              +---+---+---+---+---+---+---+---+
    Thr 3     | 3 | 7 |11 |15 |19 |23 |27 |31 |
              +---+---+---+---+---+---+---+---+
```

## 4. TV Layout 的两个 Slice

TV Layout 可以理解为两部分的组合：

```cpp
Layout<
    Shape <Thr_Shape, Val_Shape>,
    Stride<Thr_Stride, Val_Stride>
>
//      +-- Thr Slice --+  +-- Val Slice --+
```

| Slice | 描述 | 含义 |
|-------|------|------|
| **Thr Slice** | 第一个维度 | Thread 的起始位置 (Register 0 的位置) |
| **Val Slice** | 第二个维度 | 每个 Register 相对起始的偏移 (所有 Thread 共用) |

### 4.1 公式分解

```cpp
offset(thread_id, value_id) = thr_offset(thread_id) + val_offset(value_id)
//                            +------ base position ----+   +-- relative offset --+
```

**类比理解：**
- **Thr Slice** = 每个员工的工位起始地址
- **Val Slice** = 工位内部抽屉的相对位置

## 5. 复杂示例：ALayout_64x16

### 5.1 Layout 定义

```cpp
using ALayout_64x16 = Layout<
    Shape <Shape <  _4, _8, _4>, Shape < _2, _2,  _2>>,
    Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>
>;
//      +------ Thr Slice ------+  +---- Val Slice ----+
//         128 线程 (4×8×4)         8 值/线程 (2×2×2)
```

### 5.2 维度拆解

| 维度 | Shape | Stride | 元素数 |
|------|-------|--------|--------|
| Thr (线程) | (4, 8, 4) | (128, 1, 16) | 4×8×4 = 128 |
| Val (值) | (2, 2, 2) | (64, 8, 512) | 2×2×2 = 8 |

**总元素数**: 128 线程 × 8 值/线程 = 1024 = 64 × 16 ✓

### 5.3 Thread 0 的 8 个 Value (Column-Major)

```cpp
// Thread 0: t0=0, t1=0, t2=0
// thr_offset = 0

// 8 个 Value 的 offset 和 (m, k) 坐标:
Val   (v0,v1,v2)   offset   m = offset%64   k = offset/64
------------------------------------------------------------
 0    (0,0,0)        0           0               0
 1    (1,0,0)       64           0               1
 2    (0,1,0)        8           8               0
 3    (1,1,0)       72           8               1
 4    (0,0,1)      512           0               8
 5    (1,0,1)      576           0               9
 6    (0,1,1)      520           8               8
 7    (1,1,1)      584           8               9
```

### 5.4 Diagram: Thread 0's 8 positions

```
64x16 matrix (Column-Major):

                k=0     k=1           k=8     k=9
              +-------+-------+     +-------+-------+
         m=0  |  V0   |  V1   | ... |  V4   |  V5   |
              | (0,0) | (0,1) |     | (0,8) | (0,9) |
              +-------+-------+     +-------+-------+

              +-------+-------+     +-------+-------+
         m=8  |  V2   |  V3   | ... |  V6   |  V7   |
              | (8,0) | (8,1) |     | (8,8) | (8,9) |
              +-------+-------+     +-------+-------+

Thread 0's 8 positions:
  - V0,V1,V4,V5 at row m=0
  - V2,V3,V6,V7 at row m=8
```

## 6. SS 模式的特殊 TV Layout

### 6.1 Thr Stride = 0

```cpp
// SS 模式: 所有线程共享相同数据
using TVLayout = Layout<
    Shape <_128, Shape <_64, _16>>,   // (Thr=128, Val=(64,16))
    Stride<  _0, Stride<  _1, _64>>   // Thr stride = 0 !
>;

// offset = thread_idx * 0 + val_m * 1 + val_k * 64
//        = val_m + val_k * 64  (与 thread 无关!)
```

### 6.2 Diagram

```
           Val: (0,0) (1,0) (2,0) ... (0,1) (1,1) ...
              +-----+-----+-----+---+-----+-----+---+
    Thr 0     |  0  |  1  |  2  |...|  64 |  65 |...|
              +-----+-----+-----+---+-----+-----+---+
    Thr 1     |  0  |  1  |  2  |...|  64 |  65 |...|  <- same!
              +-----+-----+-----+---+-----+-----+---+
    Thr 2     |  0  |  1  |  2  |...|  64 |  65 |...|  <- same!
              +-----+-----+-----+---+-----+-----+---+

All threads see the same offset -> access via same SMEM descriptor
```

## 7. TV Layout 的伪代码实现

```cpp
// 多维 TV Layout 函数
int tv_layout(int thread_id, int value_id) {
    // Step 1: 分解 thread_id 到多维坐标 (t0, t1, t2)
    int t0 = (thread_id / 1) % 4;
    int t1 = (thread_id / 4) % 8;
    int t2 = (thread_id / 32) % 4;
    
    // Step 2: 分解 value_id 到多维坐标 (v0, v1, v2)
    int v0 = (value_id / 1) % 2;
    int v1 = (value_id / 2) % 2;
    int v2 = (value_id / 4) % 2;
    
    // Step 3: 计算 offset
    int thr_offset = t0 * 128 + t1 * 1 + t2 * 16;
    int val_offset = v0 * 64  + v1 * 8 + v2 * 512;
    
    return thr_offset + val_offset;
}
```

## 8. CuTe 中的 Layout 调用

```cpp
auto layout = ALayout_64x16{};

// 方式 1: 直接调用
int offset = layout(thread_id, value_id);

// 方式 2: 分开调用
int offset = layout(make_coord(t0, t1, t2), make_coord(v0, v1, v2));

// 方式 3: 获取特定线程的子 layout
auto thr_layout = layout(thread_id, _);  // 固定 thread，返回 Val 的 layout
int offset = thr_layout(value_id);
```

## 9. 总结

```
TV Layout 核心理解:

  Layout = 坐标到整数的映射函数
  
  (i, j, k, ...) --> offset

TV Layout:

  (thread_id, value_id) --> offset
  
  Thr Slice: 决定"从哪开始"
  Val Slice: 决定"怎么分布"
  
  最终 offset = 起始 + 相对偏移
```

| 特性 | SS 模式 | RS 模式 |
|------|---------|---------|
| Thr Stride | 0 | ≠ 0 |
| 每线程数据 | 共享整块 SMEM | 各自不同位置 |
| 数据位置 | SMEM | Register |
| 复杂度 | 低 | 高 |
