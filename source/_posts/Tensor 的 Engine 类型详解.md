---
title: CuTe Tensor 的 Engine 类型详解
date: 2024-12-24 12:00:00
tags:
  - CUTLASS
  - CuTe
  - Tensor
  - Engine
categories:
  - GPU Computing
---

本文解析 CuTe 中 Tensor 的 Engine 类型，理解指针 Engine、ArrayEngine 和 ViewEngine 的区别与用途。

<!-- more -->

## 1. Tensor 模板结构

```cpp
template <class Engine, class Layout>
struct Tensor {
    Engine engine_;   // 数据存储/访问方式
    Layout layout_;   // 形状和步长
};

// TA 就是 Engine 类型
Tensor<TA, ALayout>
//     ↑
//   Engine: 决定数据如何存储/访问
```

------

## Engine 的三种主要类型

### 1. 指针 Engine (指向外部内存)

```cpp
// 指向 GMEM
Tensor<float*, Layout>           // 可写
Tensor<float const*, Layout>     // 只读

// 指向 SMEM
Tensor<smem_ptr<half_t>, Layout>
```

### 2. ArrayEngine (寄存器存储)

```cpp
// 数据存储在寄存器数组中
Tensor<ArrayEngine<float, 32>, Layout>
//                  ↑     ↑
//                类型   数量

// 使用
auto tensor = make_tensor<float>(make_shape(4, 8));  // 32 个 float
// tensor.data() 返回 ArrayEngine<float, 32>&
// 实际是 float[32] 在寄存器中
```

### 3. ViewEngine (视图/Descriptor)

```cpp
// 不拥有数据，只是视图
Tensor<ViewEngine<Tensor<...>>, Layout>

// SMEM Descriptor (WGMMA SS 模式)
Tensor<smem_desc<half_t>, Layout>
// 64-bit descriptor 指向 SMEM
```

------

## Engine 决定 .data() 返回什么

```cpp
Tensor<float*, Layout> t1;
t1.data();  // 返回 float*

Tensor<ArrayEngine<float, 8>, Layout> t2;
t2.data();  // 返回 float(&)[8]，寄存器数组引用

Tensor<smem_desc<half_t>, Layout> t3;
t3.data();  // 返回 uint64_t，SMEM descriptor
```

------

## MMA 中的 Engine 类型

### 累加器 (C/D)

```cpp
// 累加器在寄存器中
using CRegisters = uint32_t[8];  // MMA_Op 定义

Tensor<ArrayEngine<uint32_t, 8>, Layout> accum;
// 8 个 32-bit 寄存器存储累加结果
```

### SS 模式 (A/B 在 SMEM)

```cpp
// A, B 用 SMEM descriptor
using ARegisters = uint64_t[1];  // 1 个 64-bit descriptor

Tensor<smem_desc<half_t>, Layout> tCrA;
// tCrA.data() 返回 64-bit SMEM descriptor
// WGMMA 硬件直接从 descriptor 指向的 SMEM 读取
```

### RS 模式 (A 在 Register)

```cpp
// A 需要加载到寄存器
using ARegisters = uint32_t[4];  // 4 个 32-bit 寄存器

Tensor<ArrayEngine<uint32_t, 4>, Layout> tCrA;
// 数据从 SMEM 复制到这 4 个寄存器
```

------

| Engine 类型        | 存储位置      | .data() 返回       | 典型用途      |
| ------------------ | ------------- | ------------------ | ------------- |
| `T*`               | GMEM/SMEM     | `T*` 指针          | 内存访问      |
| `ArrayEngine<T,N>` | Register      | `T(&)[N]` 数组引用 | 累加器        |
| `smem_desc<T>`     | - (描述 SMEM) | `uint64_t`         | WGMMA SS 模式 |
| `ViewEngine<...>`  | 取决于内部    | 视图               | 不拷贝数据    |

## 对比不同创建方式

-------

```cpp
// 1. 寄存器 Tensor (无指针参数)
auto reg_tensor = make_tensor<float>(make_shape(4, 8));
// Engine = ArrayEngine<float, 32>
// 位置 = 寄存器/local memory (线程私有)

// 2. GMEM Tensor (有指针参数)
float* gmem_ptr = ...;
auto gmem_tensor = make_tensor(gmem_ptr, make_shape(4, 8));
// Engine = float*
// 位置 = Global Memory (指针指向的位置)

// 3. SMEM Tensor (有 SMEM 指针)
__shared__ float smem[32];
auto smem_tensor = make_tensor(make_smem_ptr(smem), make_shape(4, 8));
// Engine = smem_ptr<float>
// 位置 = Shared Memory
```

