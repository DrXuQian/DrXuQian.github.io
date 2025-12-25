---
title: "0x0E CuTe Layout 分割操作：logical_divide 与 zipped_divide 详解"
date: 2024-12-24
categories:
  - CUTLASS
tags:
  - CUTLASS
  - CuTe
  - Layout
  - logical_divide
  - zipped_divide
---

本文详细解析 CuTe 的 Layout 分割操作：`logical_divide` 和 `zipped_divide`。理解这两个操作对于掌握 CUTLASS 中的 tiled tensor 操作和 Thread-Value Layout 至关重要。

<!-- more -->

> **示例代码**:
> - [0x0E_logical_divide.cu](https://github.com/DrXuQian/cute-examples/blob/main/0x0E_logical_divide.cu)
> - [0x0E_zipped_divide.cu](https://github.com/DrXuQian/cute-examples/blob/main/0x0E_zipped_divide.cu)

> **Key Points**
> 1. **logical_divide**: Hierarchically divides layout, keeping original offset mapping
> 2. **zipped_divide**: Divides and zips tile dimensions together for easier iteration
> 3. **Shape transformation**: Both change shape but differ in stride organization
> 4. **Use case**: `zipped_divide` is commonly used for thread-value decomposition in MMA

## 1. Original Layout

Let's start with a simple 8x8 layout:

```cpp
auto layout_2d = make_layout(make_shape(Int<8>{}, Int<8>{}));
// Layout: (_8,_8):(_1,_8)
```

**Visualization** (click to view PDF): [step0_original_8x8.pdf](/assets/cute-layout-divide/step0_original_8x8.pdf)

```
Original 2D layout: (_8,_8):(_1,_8)
       0    1    2    3    4    5    6    7
    +----+----+----+----+----+----+----+----+
 0  |  0 |  8 | 16 | 24 | 32 | 40 | 48 | 56 |
    +----+----+----+----+----+----+----+----+
 1  |  1 |  9 | 17 | 25 | 33 | 41 | 49 | 57 |
    +----+----+----+----+----+----+----+----+
 2  |  2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
    +----+----+----+----+----+----+----+----+
 3  |  3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
    +----+----+----+----+----+----+----+----+
 4  |  4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
    +----+----+----+----+----+----+----+----+
 5  |  5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
    +----+----+----+----+----+----+----+----+
 6  |  6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
    +----+----+----+----+----+----+----+----+
 7  |  7 | 15 | 23 | 31 | 39 | 47 | 55 | 63 |
    +----+----+----+----+----+----+----+----+
```

The offset formula is: `offset = m * 1 + n * 8`

## 2. Tile Definition

We define a tile to divide the layout:

```cpp
auto tile_2d = make_tile(make_layout(Int<4>{}),   // M: groups of 4
                         make_layout(Int<2>{}));  // N: groups of 2
// Tile: (_4:_1, _2:_1)
```

This means:
- Divide M dimension into groups of 4 rows
- Divide N dimension into groups of 2 columns
- Result: 2x4 = 8 tiles covering the 8x8 matrix

## 3. logical_divide

```cpp
auto divided_2d = logical_divide(layout_2d, tile_2d);
// Result: ((_4,_2),(_2,_4)):((_1,_4),(_8,_16))
```

**Visualization**: [step1_logical_divide.pdf](/assets/cute-layout-divide/step1_logical_divide.pdf)

### Shape Analysis

| Dimension | Meaning |
|-----------|---------|
| `(_4,_2)` | First dim: 4 rows per tile, 2 tile rows |
| `(_2,_4)` | Second dim: 2 cols per tile, 4 tile cols |

### Stride Analysis

| Stride | Meaning |
|--------|---------|
| `_1` | Adjacent rows within tile (original M stride) |
| `_4` | Jump between tile rows (4 rows apart) |
| `_8` | Adjacent columns within tile (original N stride) |
| `_16` | Jump between tile columns (2 cols * 8 stride) |

### Key Property: Original Offset Preserved

```
After logical_divide: ((_4,_2),(_2,_4)):((_1,_4),(_8,_16))
       0    1    2    3    4    5    6    7
    +----+----+----+----+----+----+----+----+
 0  |  0 |  8 | 16 | 24 | 32 | 40 | 48 | 56 |
    +----+----+----+----+----+----+----+----+
 1  |  1 |  9 | 17 | 25 | 33 | 41 | 49 | 57 |
    +----+----+----+----+----+----+----+----+
 ...
```

The offset mapping is **identical** to the original layout! `logical_divide` only restructures the logical view, not the physical layout.

## 4. zipped_divide

```cpp
auto zipped = zipped_divide(layout_2d, tile_2d);
// Result: ((_4,_2),(_2,_4)):((_1,_8),(_4,_16))
```

**Visualization**: [step2_zipped_divide.pdf](/assets/cute-layout-divide/step2_zipped_divide.pdf)

### Different Stride Organization

Compare the strides:

| Operation | Shape | Stride |
|-----------|-------|--------|
| logical_divide | `((_4,_2),(_2,_4))` | `((_1,_4),(_8,_16))` |
| zipped_divide | `((_4,_2),(_2,_4))` | `((_1,_8),(_4,_16))` |

Notice: `zipped_divide` **zips** the inner tile dimensions together!

### Offset Mapping Changes

```
After zipped_divide: ((_4,_2),(_2,_4)):((_1,_8),(_4,_16))
       0    1    2    3    4    5    6    7
    +----+----+----+----+----+----+----+----+
 0  |  0 |  4 | 16 | 20 | 32 | 36 | 48 | 52 |
    +----+----+----+----+----+----+----+----+
 1  |  1 |  5 | 17 | 21 | 33 | 37 | 49 | 53 |
    +----+----+----+----+----+----+----+----+
 2  |  2 |  6 | 18 | 22 | 34 | 38 | 50 | 54 |
    +----+----+----+----+----+----+----+----+
 3  |  3 |  7 | 19 | 23 | 35 | 39 | 51 | 55 |
    +----+----+----+----+----+----+----+----+
 4  |  8 | 12 | 24 | 28 | 40 | 44 | 56 | 60 |
    +----+----+----+----+----+----+----+----+
 5  |  9 | 13 | 25 | 29 | 41 | 45 | 57 | 61 |
    +----+----+----+----+----+----+----+----+
 6  | 10 | 14 | 26 | 30 | 42 | 46 | 58 | 62 |
    +----+----+----+----+----+----+----+----+
 7  | 11 | 15 | 27 | 31 | 43 | 47 | 59 | 63 |
    +----+----+----+----+----+----+----+----+
```

The offsets are now **reordered** - elements within each tile are now contiguous!

## 5. Understanding the Difference

### Visual Comparison

**logical_divide** - Hierarchical grouping:
```
First 4x2 tile:     Second 4x2 tile (offset by 16):
[0, 8]              [16, 24]
[1, 9]              [17, 25]
[2, 10]             [18, 26]
[3, 11]             [19, 27]
```

**zipped_divide** - Contiguous tiles:
```
First 4x2 tile:     Second 4x2 tile (offset by 16):
[0, 4]              [16, 20]
[1, 5]              [17, 21]
[2, 6]              [18, 22]
[3, 7]              [19, 23]
```

### Mathematical Interpretation

For coordinate `(m, n)` with tile size `(TileM, TileN)`:

**logical_divide**:
- Inner coord: `(m % TileM, n % TileN)`
- Outer coord (tile index): `(m / TileM, n / TileN)`
- Offset uses original strides

**zipped_divide**:
- Same hierarchical decomposition
- But strides are adjusted so each tile's elements are contiguous
- Tile index effectively becomes the major index

## 6. When to Use Which

| Use Case | Recommended |
|----------|-------------|
| Preserving memory access pattern | `logical_divide` |
| Thread-Value decomposition | `zipped_divide` |
| Tiled iteration with contiguous tiles | `zipped_divide` |
| Hierarchical view without data movement | `logical_divide` |

In CUTLASS, `zipped_divide` is heavily used for:
1. Converting tensor layouts to Thread-Value (TV) format
2. Creating fragment views for MMA operations
3. Iterating over tiles in a regular pattern

## 7. Example Code

```cpp
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
    // Original 8x8 layout
    auto layout_2d = make_layout(make_shape(Int<8>{}, Int<8>{}));

    // Tile: 4x2
    auto tile_2d = make_tile(make_layout(Int<4>{}),
                             make_layout(Int<2>{}));

    // logical_divide - preserves offsets
    auto divided = logical_divide(layout_2d, tile_2d);
    print("logical_divide: "); print(divided); print("\n");

    // zipped_divide - contiguous tiles
    auto zipped = zipped_divide(layout_2d, tile_2d);
    print("zipped_divide: "); print(zipped); print("\n");

    return 0;
}
```

## 8. Summary

| Property | logical_divide | zipped_divide |
|----------|----------------|---------------|
| Shape | Same | Same |
| Stride | Hierarchical (original) | Zipped (reordered) |
| Offset mapping | Preserved | Changed |
| Use case | View transformation | Tile iteration |

Both operations are fundamental to CuTe's layout algebra and enable efficient tiled tensor operations. Understanding their differences is key to mastering CUTLASS's tensor abstractions.

## References

- [CUTLASS CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- Source: `cute-examples/logical_divide.cu`
