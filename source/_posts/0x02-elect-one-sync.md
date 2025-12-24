---
title: "0x02 elect_one_sync and Warp-Level Leader Election"
date: 2024-12-24
categories: [CUDA, PTX, Hopper]
tags: [elect, warp, synchronization, leader]
---

This article compares `elect.sync` instruction with the traditional `threadIdx % 32 == 0` method for warp-level leader election.

<!-- more -->

## 1. Introduction

In CUDA programming, we often need to select a "leader" thread in a warp to execute certain operations (like barrier arrive, initialization, etc.). This article compares `elect.sync` instruction with the traditional `threadIdx % 32 == 0` method.

## 2. 两种方法对比

```cpp
// 方法 1: 简单判断 (可能有问题!)
bool is_leader = (threadIdx.x % 32) == 0;

// 方法 2: elect.sync (SM90+, 更健壮!)
bool is_leader = cute::elect_one_sync();
```

## 3. 核心问题：部分线程不活跃

### 3.1 问题场景

```cpp
if (some_condition) {  // 假设只有部分线程满足条件
    // Lane 0 可能不在这里！
    
    if (threadIdx.x % 32 == 0) {  // ❌ Lane 0 不活跃，没人执行!
        do_something();
    }
}
```

### 3.2 图示

```
Warp 32 threads:

Lane:  0   1   2   3   4   5   6   7  ...  31
       x   o   o   x   o   o   o   x  ...  o
       |   |
       |   +-- active
       +------ inactive (Lane 0 not in this branch!)

threadIdx % 32 == 0: no one satisfies!
elect.sync: picks from active threads (e.g., Lane 1)
```

## 4. elect.sync PTX 指令

### 4.1 CuTe 实现

```cpp
CUTE_HOST_DEVICE uint32_t elect_one_sync()
{
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"  // mask = 0xFFFFFFFF
        "@%%px mov.s32 %1, 1;\n"            // 被选中的线程设 pred = 1
        "     mov.s32 %0, %%rx;\n"          // 返回被选中的 lane id
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
#elif defined(__CUDA_ARCH__)
    return (threadIdx.x % 32) == 0;  // Fallback
#else
    return true;
#endif
}
```

### 4.2 输出含义

| 输出 | 含义 |
|------|------|
| `%%rx` | 被选中的 lane ID |
| `%%px` | 当前线程是否被选中 (predicate) |

## 5. 对比示例

### 场景 1：所有线程活跃

```
Lane:     0   1   2   3   ...  31
活跃:     ✓   ✓   ✓   ✓   ...  ✓

threadIdx % 32 == 0:  Lane 0 执行 ✓
elect.sync:           Lane 0 执行 ✓  (选最小活跃 lane)

结果相同 ✓
```

### 场景 2：Lane 0 不活跃

```
Lane:     0   1   2   3   ...  31
活跃:     ✗   ✓   ✓   ✓   ...  ✓

threadIdx % 32 == 0:  没人执行! ✗
elect.sync:           Lane 1 执行 ✓  (选最小活跃 lane)

elect.sync 更健壮!
```

### 场景 3：只有少数线程活跃

```
Lane:     0   1   2   3   4   5   ...  31
活跃:     ✗   ✗   ✗   ✗   ✓   ✓   ...  ✗

threadIdx % 32 == 0:  没人执行! ✗
elect.sync:           Lane 4 执行 ✓

elect.sync 总能选出一个 leader!
```

## 6. 实际使用场景

### 6.1 Barrier 初始化

```cpp
void initialize_barrier_array_aligned(T ptr, int arv_cnt) {
    // 只需要一个线程执行初始化
    if (cute::elect_one_sync()) {
        for (int i = 0; i < Stages; i++) {
            ptr[i].init(arv_cnt);
        }
    }
}
```

### 6.2 条件分支中的 Leader 选举

```cpp
if (should_do_work) {
    // 这个分支里可能只有部分线程
    // Lane 0 可能不在这里
    
    if (cute::elect_one_sync()) {
        // 保证有一个线程执行
        barrier.arrive();
    }
}
```

### 6.3 TMA 操作

```cpp
if (cute::elect_one_sync()) {
    // 只有一个线程发起 TMA
    tma_load_multicast(...);
}
```

## 7. 额外好处：隐式同步

```cpp
elect.sync %%rx|%%px, mask;
//    ↑
//   sync! 参与的线程会同步
```

`elect.sync` 自带 warp 同步，确保：
1. 所有活跃线程到达这个点
2. 选举结果对所有线程一致
3. 避免竞争条件

## 8. Fallback 行为

```cpp
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
    // SM90+: 使用 elect.sync 指令
    asm volatile("elect.sync ...");
    return pred;
#elif defined(__CUDA_ARCH__)
    // 旧架构: 没有 elect.sync，假设所有线程活跃
    return (threadIdx.x % 32) == 0;
#else
    return true;
#endif
```

**说明：**
- `elect.sync` 是 SM90 (Hopper) 新增指令
- 旧架构只能用 `threadIdx % 32 == 0`
- 旧代码通常保证所有线程都活跃，所以可以工作

## 9. 常见使用模式

### 9.1 Warp 级别选举

```cpp
// 每个 warp 选一个 leader
if (cute::elect_one_sync()) {
    // 一个 warp 一个线程执行
}
```

### 9.2 Warpgroup 级别选举 (4 warps)

```cpp
// 只在 warp 0 中选举
if (warp_idx == 0 && cute::elect_one_sync()) {
    // 一个 warpgroup 一个线程执行
}
```

### 9.3 Block 级别选举

```cpp
// 只在 warp 0 中选举
if (threadIdx.x / 32 == 0 && cute::elect_one_sync()) {
    // 一个 block 一个线程执行
}
```

## 10. 性能考虑

| 方面 | `threadIdx % 32 == 0` | `elect.sync` |
|------|----------------------|--------------|
| 指令数 | 1 条比较 | ~3 条指令 |
| 同步 | 无 | 隐式 warp sync |
| 健壮性 | 低 | 高 |
| 架构支持 | 所有 | SM90+ |

**结论：性能差异微小，`elect.sync` 的健壮性优势远大于微小的性能开销。**

## 11. 总结

| 特性 | `threadIdx % 32 == 0` | `elect.sync` |
|------|----------------------|--------------|
| Lane 0 不活跃时 | ❌ 没人执行 | ✓ 选其他活跃线程 |
| 部分线程活跃 | ❌ 可能没人执行 | ✓ 总能选一个 |
| 隐式同步 | ❌ 无 | ✓ 有 |
| 架构要求 | 任意 | SM90+ |
| 适用场景 | 保证全活跃 | 通用 |

**核心理解：`elect.sync` 从当前活跃线程中选举一个 leader，即使 Lane 0 不活跃也能正常工作，比硬编码 `threadIdx % 32 == 0` 更健壮。在 SM90+ 架构上应优先使用 `elect_one_sync()`。**
