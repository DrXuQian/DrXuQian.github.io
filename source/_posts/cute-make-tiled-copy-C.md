---
title: CuTe make_tiled_copy_C 深度解析：从 MMA 到 Shared Memory
date: 2025-01-13 22:30:00
tags:
  - CUDA
  - CUTLASS
  - CuTe
  - GPU
categories:
  - CUTLASS
---

# CuTe make_tiled_copy_C 深度解析：从 MMA 到 Shared Memory

## 1. 典型使用场景

在 GEMM kernel 的 epilogue 阶段，需要将 MMA 计算结果从寄存器写入 shared memory：

```cpp
extern __shared__ T shm_data[];

T *Cshm = shm_data;
auto sC = make_tensor(make_smem_ptr(Cshm), SmemLayoutC{});
using SmemLayoutC = decltype(make_layout(make_shape(Int<32>{}, Int<32>{})));
using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, cute::half_t>;

// 从 tiled mma 延伸出 r2s tiled copy
auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(thread_idx);
auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);   // 源：寄存器中的 C fragment
auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // 目的：shared memory

cute::copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);

__syncthreads();
```

## 2. 核心组件解析

### 2.1 `make_tiled_copy_C` 的作用

`make_tiled_copy_C` 从 TiledMMA 的 C 输出 layout 创建一个兼容的 TiledCopy：

```cpp
// copy_atom.hpp:442-446
template <class... CArgs, class... MArgs>
auto make_tiled_copy_C(Copy_Atom<CArgs...> const& copy_atom,
                       TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom, mma.get_layoutC_TV(),
                              make_shape(tile_size<0>(mma), tile_size<1>(mma)));
}
```

**关键点**：
- 使用 MMA 的 `layoutC_TV` 作为 TiledCopy 的 TV layout
- 这确保了 copy 操作与 MMA 的线程-value 映射一致

### 2.2 TiledMMA 的 `layoutC_TV` 结构

```cpp
// mma_atom.hpp:399-411
auto get_layoutC_TV() const {
  auto ref_C = make_layout(make_shape(tile_size_mnk<0>(), tile_size_mnk<1>()));
  // (thr_idx, val) -> (M, N)
  return thrfrg_C(ref_C).compose(thridx_2_thrid, _);
}
```

返回的 layout 结构为：
```
(thread_idx, (FrgV, (RestM, RestN))) -> (M, N)
```

其中：
- `thread_idx`：线程索引（0 到 NumThreads-1）
- `FrgV`：单个 MMA atom 每线程的 values（如 `(2,2)` = 4 个）
- `RestM/RestN`：M/N 方向的 tile 重复次数

### 2.3 `retile_S(tCrC)` - 源寄存器重组

`retile_S` 将 MMA 输出的寄存器 tensor 重组为 Copy Atom 期望的格式：

```cpp
// 输入：MMA 的嵌套布局
tCrC: ((2,2), (RestM, RestN))  // FrgV 是嵌套的 (2,2)

// 输出：Copy 期望的扁平格式
tCrC_r2s: (R2S, R2S_V, MMA_M, MMA_N)  // R2S 是每次 copy 的元素数
```

**为什么需要 retile？**

MMA 和 Copy 的 value 组织方式不同：
- MMA 按照硬件指令输出排列（如 `mma.sync` 的寄存器布局）
- Copy 按照存储指令的操作粒度分组（如 `stmatrix` 一次存 8 个 FP16）

`retile` 重新组织 values，使其匹配 copy atom 的迭代模式。

### 2.4 `partition_D(sC)` - 目的地址分区

`partition_D` 为每个线程计算其在 shared memory 中的写入位置：

```cpp
// 实现原理 (copy_atom.hpp:375-383)
auto partition_D(Tensor&& tensor) {
  auto thr_tensor = tidfrg_D(tensor);  // 应用 TV layout
  return thr_tensor(thr_idx_, _);       // 提取当前线程的分区
}
```

**计算过程**：

1. **SmemLayoutC** 定义了 `(m, n) -> smem_index` 的映射
2. **TV layout** 定义了 `(thread_idx, value_idx) -> (m, n)` 的映射
3. **组合**：`SmemLayoutC.compose(TV_Layout)` 得到 `(thread_idx, value_idx) -> smem_index`
4. **提取**：`partition_D` 提取当前线程的所有 value 的 smem 地址

### 2.5 `cute::copy` 的执行

```cpp
// copy.hpp:224-227
CUTE_UNROLL
for (int i = 0; i < size<1>(dst_c); ++i) {
  copy_atom.call(src_c(_,i), dst_c(_,i));
}
```

对于 `(R2S, R2S_V)` 格式的 tensor：
- 外层循环 `R2S_V` 次
- 每次调用 `copy_atom.call` 处理 `R2S` 个元素

## 3. 数据流完整示例

假设配置：
- TiledMMA: 128 线程，每线程 16 个 values（`((2,2), (2,2))`）
- SmemLayoutC: 64×64
- R2SCopyAtom: `UniversalCopy<int>`（一次 copy 2 个 FP16）

### 3.1 MMA 输出

```
每线程 tCrC: ((2,2), (2,2)) = 16 values
              ↑       ↑
           FrgV   (RestM, RestN)

物理布局：values 按 MMA 指令输出排列
```

### 3.2 retile_S 后

```
tCrC_r2s: (2, 8)
           ↑  ↑
    atom_vals  iterations

含义：每次 copy 2 个元素，循环 8 次
```

### 3.3 partition_D 后

```
tCsC_r2s: (2, 8)
           ↑  ↑
      每次写入  迭代次数

每个位置包含一个 smem 地址
```

### 3.4 copy 执行

```cpp
// 伪代码
for (int i = 0; i < 8; ++i) {
    // 从寄存器读取 2 个 FP16
    auto src = tCrC_r2s(_, i);  // 2 elements
    // 写入 shared memory 对应位置
    auto dst = tCsC_r2s(_, i);  // 2 smem addresses
    copy_atom.call(src, dst);   // 一条 store 指令
}
```

## 4. Copy Atom 选择

### 4.1 常见 Copy Atom 类型

| Copy Atom | 每次元素数 | 适用场景 |
|-----------|-----------|---------|
| `UniversalCopy<int>` | 2 FP16 | 通用，32-bit store |
| `UniversalCopy<int2>` | 4 FP16 | 64-bit store |
| `UniversalCopy<int4>` | 8 FP16 | 128-bit store |
| `SM90_U16x8_STSM_T` | 8 FP16 | stmatrix.x4.trans |
| `SM90_U32x4_STSM_N` | 8 FP16 | stmatrix.x4 |

### 4.2 选择依据

```cpp
// sm90_common.inl:42-65
if constexpr (sizeof(ElementD) == 2) {  // FP16/BF16
    if constexpr (EpilogueTile_N % 16 == 0) {
        return SM90_U16x8_STSM_T{};  // 使用 stmatrix
    }
}
// fallback to auto-vectorizing
return AutoVectorizingCopy{};
```

## 5. 与 SmemLayout 的配合

### 5.1 Swizzle 的影响

如果 `SmemLayoutC` 包含 swizzle（用于避免 bank conflict）：

```cpp
using SmemLayoutC = decltype(
    composition(Swizzle<3,3,3>{},  // swizzle 参数
                Layout<Shape<_64, _64>>{}));
```

`partition_D` 会自动处理 swizzle，返回正确的物理地址：

```cpp
auto sC = make_tensor(make_smem_ptr(Cshm), SmemLayoutC{});
auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);
// tCsC_r2s 的地址已经包含 swizzle 变换
```

### 5.2 Bank Conflict Free 的关键

`make_tiled_copy_C` 确保：
1. 每个 warp 内的线程写入不同的 bank
2. 写入模式与 SmemLayout 的 swizzle 匹配

这是因为 `layoutC_TV` 继承自 MMA 的线程布局，而 MMA 本身就是为 bank conflict free 设计的。

## 6. 常见问题

### 6.1 为什么不能直接用 `cute::copy(tCrC, sC)`？

直接 copy 会导致：
1. **线程映射错误**：不知道哪个线程负责哪些元素
2. **地址计算错误**：不知道每个 value 应该写到 smem 的哪个位置
3. **无法利用向量化**：无法使用 stmatrix 等高效指令

### 6.2 `retile_S` vs `partition_S`

- `retile_S`：重组已有 tensor 的 layout（不改变数据）
- `partition_S`：从一个大 tensor 中提取当前线程的分区

对于 r2s copy：
- 源是寄存器，已经按线程分区 → 使用 `retile_S`
- 目的是 shared memory，需要分区 → 使用 `partition_D`

### 6.3 `make_tiled_copy_C` vs `make_tiled_copy_C_atom`

```cpp
// make_tiled_copy_C：使用完整的 MMA tile
auto tiled_copy = make_tiled_copy_C(copy_atom, mma);

// make_tiled_copy_C_atom：使用最小的 atom 级别
auto tiled_copy_atom = make_tiled_copy_C_atom(copy_atom, mma);
```

`make_tiled_copy_C_atom` 用于 pipelined epilogue，每次只处理一个 atom 大小的数据块。

## 7. 总结

| 步骤 | 函数 | 作用 |
|------|------|------|
| 1 | `make_tiled_copy_C` | 从 MMA 创建兼容的 TiledCopy |
| 2 | `get_slice(idx)` | 获取当前线程的 ThrCopy |
| 3 | `retile_S(tCrC)` | 重组寄存器 layout 为 copy 格式 |
| 4 | `partition_D(sC)` | 计算 smem 写入地址 |
| 5 | `cute::copy` | 执行实际的数据传输 |

**核心要点**：
1. `make_tiled_copy_C` 保证 copy 与 MMA 的线程-value 映射一致
2. `retile_S` 解决 MMA 和 Copy 的 value 组织方式不同的问题
3. `partition_D` 计算考虑了 swizzle 的 smem 地址
4. 整个流程确保 bank conflict free 和高效向量化

## 参考资料

- CUTLASS 源码: `include/cute/atom/copy_atom.hpp`
- CUTLASS 源码: `include/cute/atom/mma_atom.hpp`
- CUTLASS 源码: `include/cute/algorithm/copy.hpp`
- [CuTe Tiled Copy - Lei Mao's Log Book](https://leimao.github.io/blog/CuTe-Tiled-Copy/)
