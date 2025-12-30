---
title: "0x15 CuTe 特殊符号：_ 与 X 的含义与用法"
date: 2024-12-30 12:00:00
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
  - Underscore
  - Slicing
  - Layout
---

本文详解 CuTe 中的两个特殊符号：`_`（下划线）和 `X`。它们是 CuTe 中最常用的通配符/占位符，理解它们对于阅读和编写 CuTe 代码至关重要。

<!-- more -->

## 1. 定义

```cpp
// include/cute/underscore.hpp

// For slicing
struct Underscore : Int<0> {};

CUTE_INLINE_CONSTANT Underscore _;

// Convenient alias
using X = Underscore;
```

**核心要点**：
- `_` 和 `X` 是完全相同的东西
- `X` 只是 `Underscore` 的类型别名
- `Underscore` 继承自 `Int<0>`，但具有特殊的类型身份

## 2. 主要用途

### 2.1 Tensor 切片（Slicing）

最常见的用法是在 Tensor 索引中作为"保留该维度"的占位符。

```cpp
// 3D Tensor: (M, N, K)
auto tensor = make_tensor(ptr, make_shape(M, N, K));

// 切片：固定第三维为 0，保留前两维
auto slice_2d = tensor(_, _, 0);  // -> (M, N) tensor

// 切片：固定第一维为 5，保留后两维
auto slice_mn = tensor(5, _, _);  // -> (N, K) tensor

// 切片：只取单个元素
auto elem = tensor(1, 2, 3);      // -> 单个值
```

**工作原理**：

```cpp
// tensor_impl.hpp 中的 operator()
template <class Coord>
decltype(auto) operator()(Coord const& coord) {
  if constexpr (has_underscore<Coord>::value) {
    // 有 _ 时返回子 tensor
    auto [sliced_layout, offset] = slice_and_offset(coord, layout());
    return make_tensor(data() + offset, sliced_layout);
  } else {
    // 无 _ 时返回元素
    return data()[layout()(coord)];
  }
}
```

### 2.2 compose 中保留维度

在 `compose` 操作中，`_` 表示"不变换该维度"。

```cpp
// Layout: ((Thr, Val), (RestM, RestN))
auto tv_tensor = tensor.compose(thrval2mn, _);
//                              ^^^^^^^^^  ^
//                              变换第一维  保留第二维
```

**实现原理**（`layout.hpp`）：

```cpp
template <class LShape, class LStride, class Tiler>
auto composition(Layout<LShape,LStride> const& lhs, Tiler const& rhs) {
  if constexpr (is_underscore<Tiler>::value) {
    return lhs;  // 直接返回原 layout，不做任何变换
  }
  // ...
}
```

### 2.3 make_tile 中的占位符

在 `make_tile` 中，`_` 表示"整个维度作为一个单位"。

```cpp
// thrfrg_C 中的用法
auto thr_tile = make_tile(_,                                    // ThrV 维度整体
                          make_tile(make_layout(size<1>(thr)),  // ThrM
                                    make_layout(size<2>(thr)))); // ThrN

// zipped_divide(tv_tensor, thr_tile) 会：
// - 第一维 (_)：保持不变，不做 divide
// - 第二维：按 (ThrM, ThrN) 进行 divide
```

## 3. slice 与 dice 函数

CuTe 提供了两个对偶函数来处理带 `_` 的坐标：

### 3.1 slice：保留 `_` 对应的元素

```cpp
// slice(pattern, tuple) -> 保留 pattern 中 _ 位置的元素
slice(make_tuple(_, 1), make_tuple(A, B))  // -> (A,)
slice(make_tuple(0, _), make_tuple(A, B))  // -> (B,)
slice(_, X)                                 // -> X
```

### 3.2 dice：保留非 `_` 对应的元素

```cpp
// dice(pattern, tuple) -> 保留 pattern 中非 _ 位置的元素
dice(make_tuple(_, 1), make_tuple(A, B))  // -> (B,)
dice(make_tuple(0, _), make_tuple(A, B))  // -> (A,)
dice(1, X)                                 // -> X
```

## 4. 实际代码示例

### 4.1 TiledCopy::tile2thrfrg

```cpp
template <class Tensor, class Ref2TrgLayout>
auto tile2thrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg) {
  // Step 1: zipped_divide
  auto atom_layout_TV = zipped_divide(TiledLayout_TV{},
                                       make_shape(AtomNumThr{}, AtomNumVal{}));

  // Step 2: compose with ref2trg, 保留 rest 维度
  auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
  //                                          ^^^^^^^  ^
  //                                          变换atom  保留rest

  // Step 3: coalesce + compose
  auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1,Shape<_1,_1>>{});
  auto tv_tensor = tensor.compose(thrval2mn, _);
  //                               ^^^^^^^^^  ^
  //                               变换tile    保留rest

  // Step 4: 展开返回
  return tv_tensor(make_coord(_,_), _);
  //               ^^^^^^^^^^^  ^
  //               保留thr,val   保留rest
}
```

### 4.2 thrfrg_C 中的使用

```cpp
// mma_atom.hpp
auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{}, _);  // ((ThrV,FrgV),(RestM,RestN))

auto thr_tile = make_tile(_,                              // ThrV 整体
                          make_tile(layout(size<1>(thr)), // ThrM
                                    layout(size<2>(thr))));// ThrN

auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
return thr_tensor(make_coord(_,_), _);  // 展开 thread 和 value 维度
```

### 4.3 ThrCopy 分片

```cpp
// copy_atom.hpp
template <class STensor>
auto partition_S(STensor&& stensor) {
  auto thr_tensor = make_tensor(stensor.data(),
                                 TiledCopy::tidfrg_S(stensor.layout()));
  return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
  //               ^^^^^^^^  ^  ^^^^^^^^^^^^^^^^^^^^^^^^^
  //               固定线程   保留val   保留所有 rest 维度
}
```

## 5. 类型检测

CuTe 提供了类型特征来检测 `_`：

```cpp
// 检测单个类型
template <class T>
struct is_underscore : false_type {};
template <>
struct is_underscore<Underscore> : true_type {};

// 检测 tuple 中是否包含 _
template <class Tuple>
using has_underscore = has_elem<Tuple, Underscore>;

// 检测 tuple 是否全是 _
template <class Tuple>
using all_underscore = all_elem<Tuple, Underscore>;
```

## 6. 为什么 `_` 继承 `Int<0>`？

```cpp
struct Underscore : Int<0> {};
```

这是一个巧妙的设计：
1. **类型区分**：`is_underscore<Underscore>` 为 true，但 `is_underscore<Int<0>>` 为 false
2. **数值兼容**：在需要数值的场合，`_` 可以当作 0 使用
3. **编译期常量**：继承 `Int<0>` 使其成为编译期整数

## 7. 总结

```
┌─────────────────────────────────────────────────────────────┐
│                    CuTe Special Symbols                      │
├─────────────────────────────────────────────────────────────┤
│  _ 和 X 是同一个东西（Underscore 类型）                       │
├─────────────────────────────────────────────────────────────┤
│  用途 1: Tensor 切片                                         │
│    tensor(_, 0, _)  ->  保留第1和第3维，固定第2维为0          │
├─────────────────────────────────────────────────────────────┤
│  用途 2: compose 保留维度                                     │
│    tensor.compose(layout, _)  ->  变换第1维，保留第2维        │
├─────────────────────────────────────────────────────────────┤
│  用途 3: make_tile 占位符                                     │
│    make_tile(_, tile)  ->  第1维整体，第2维按 tile 划分       │
├─────────────────────────────────────────────────────────────┤
│  相关函数:                                                    │
│    slice(pattern, tuple)  ->  保留 _ 位置的元素               │
│    dice(pattern, tuple)   ->  保留非 _ 位置的元素             │
└─────────────────────────────────────────────────────────────┘
```

### 记忆口诀

- **`_` = "保持不变"**：无论在切片、compose 还是 make_tile 中
- **`X` = `_`**：只是换个写法，语义完全相同
- **`slice` 保留 `_`**：slice 的 s 可以联想 "save underscore"
- **`dice` 丢弃 `_`**：dice 的 d 可以联想 "drop underscore"

## References

- `include/cute/underscore.hpp` - Underscore 定义
- `include/cute/tensor_impl.hpp` - Tensor operator() 实现
- `include/cute/layout.hpp` - composition 函数
- `include/cute/atom/copy_atom.hpp` - tile2thrfrg 用法示例
- `include/cute/atom/mma_atom.hpp` - thrfrg_C 用法示例
