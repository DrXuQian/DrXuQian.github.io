---
title: CUTLASS SM90 Pipeline and mbarrier Deep Dive
date: 2024-12-23
categories:
  - CUTLASS
tags:
  - CUTLASS
  - Pipeline
  - mbarrier
  - PTX
  - SM90
---

This article provides a deep dive into the CUTLASS SM90 Pipeline mechanism and how it maps to the underlying mbarrier PTX instructions. All code references are from the official NVIDIA CUTLASS repository.

<!-- more -->

## 1. Introduction

NVIDIA Hopper (SM90) introduces hardware-accelerated asynchronous data movement through TMA (Tensor Memory Accelerator) and a new barrier synchronization mechanism called `mbarrier`. CUTLASS provides a high-level Pipeline abstraction that leverages these features for efficient producer-consumer synchronization.

**Key Source Files:**
- [sm90_pipeline.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/pipeline/sm90_pipeline.hpp) - Pipeline implementation
- [barrier.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/barrier.h) - Barrier primitives

## 2. Core Data Structures

### 2.1 PipelineState

The `PipelineState` structure tracks the current position in the circular buffer:

```cpp
// Source: include/cutlass/pipeline/sm90_pipeline.hpp:170-250
template<uint32_t Stages_>
struct PipelineState {
  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;        // Current stage index (0 to Stages-1)
  uint32_t phase_ = 0;   // Current phase (0 or 1), flips when wrapping
  uint32_t count_ = 0;   // Total iteration count

  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;  // Flip phase when wrapping around
      }
    }
  }
};
```

The **phase bit** is crucial for barrier synchronization - it toggles each time we complete a full cycle through all stages, allowing the barrier to distinguish between different iterations.

### 2.2 PipelineTmaAsync Class

This is the main Pipeline class for TMA-based asynchronous loads:

```cpp
// Source: include/cutlass/pipeline/sm90_pipeline.hpp:270-299
template <int Stages_>
class PipelineTmaAsync {
public:
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];   // Signals "data ready"
    EmptyBarrier empty_barrier_[Stages]; // Signals "space available"
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0;
    uint32_t num_producers = 1;
  };
  // ...
};
```

**Key Design Points:**
- **Dual Barrier Architecture**: Each pipeline stage has two barriers - `FullBarrier` (data ready) and `EmptyBarrier` (space available)
- **ClusterTransactionBarrier**: The `FullBarrier` type supports transaction-based completion (bytes transferred)
- **ClusterBarrier**: The `EmptyBarrier` type uses traditional arrival counting

## 3. Barrier Types and PTX Instructions

### 3.1 ClusterBarrier

The `ClusterBarrier` provides cluster-wide arrive-wait synchronization:

```cpp
// Source: include/cutlass/arch/barrier.h:341-532
struct ClusterBarrier {
  using ValueType = uint64_t;

protected:
  ValueType barrier_;  // 64-bit mbarrier object in SMEM

public:
  // Initialize barrier with expected arrival count
  CUTLASS_HOST_DEVICE
  static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%1], %0;"
        : : "r"(arrive_count), "r"(smem_addr));
  }

  // Blocking wait with spin loop
  CUTLASS_HOST_DEVICE
  static void wait(ValueType const* smem_ptr, uint32_t phase) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t ticks = 0x989680;  // ~10M cycles timeout before retry
    asm volatile(
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@P1 bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:"
        : : "r"(smem_addr), "r"(phase), "r"(ticks));
  }

  // Non-blocking try wait
  CUTLASS_HOST_DEVICE
  static bool try_wait(ValueType const* smem_ptr, uint32_t phase) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t waitComplete;
    asm volatile(
        ".reg .pred P1;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.b32 %0, 1, 0, P1;"
        : "=r"(waitComplete) : "r"(smem_addr), "r"(phase));
    return static_cast<bool>(waitComplete);
  }

  // Local CTA arrive
  CUTLASS_HOST_DEVICE
  static void arrive(ValueType const* smem_ptr) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        : : "r"(smem_addr));
  }

  // Remote cluster arrive (to another CTA's SMEM)
  CUTLASS_HOST_DEVICE
  static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    if (pred) {
      asm volatile(
          ".reg .b32 remAddr32;\n"
          "mapa.shared::cluster.u32 remAddr32, %0, %1;\n"
          "mbarrier.arrive.shared::cluster.b64 _, [remAddr32];"
          : : "r"(smem_addr), "r"(cta_id));
    }
  }
};
```

### 3.2 ClusterTransactionBarrier

This extends `ClusterBarrier` with transaction byte counting, essential for TMA operations:

```cpp
// Source: include/cutlass/arch/barrier.h:538-693
struct ClusterTransactionBarrier : public ClusterBarrier {

  // Arrive + set expected transaction bytes
  CUTLASS_HOST_DEVICE
  static void arrive_and_expect_tx(ValueType const* smem_ptr, uint32_t transaction_bytes) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;"
        : : "r"(transaction_bytes), "r"(smem_addr));
  }

  // Set expected bytes without arrive (for additional TMA loads)
  CUTLASS_HOST_DEVICE
  static void expect_transaction(ValueType const* smem_ptr, uint32_t transaction_bytes) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "mbarrier.expect_tx.shared::cta.b64 [%1], %0;"
        : : "r"(transaction_bytes), "r"(smem_addr));
  }

  // Complete transaction (decrement pending bytes) - called by TMA hardware
  CUTLASS_HOST_DEVICE
  static void complete_transaction(
      ValueType const* smem_ptr, uint32_t dst_cta_id,
      uint32_t transaction_bytes, uint32_t pred = 1) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    smem_addr = cute::set_block_rank(smem_addr, dst_cta_id);
    asm volatile(
        ".reg .pred p;\n"
        "setp.eq.u32 p, %2, 1;\n"
        "@p mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64 [%1], %0;"
        : : "r"(transaction_bytes), "r"(smem_addr), "r"(pred));
  }
};
```

## 4. Pipeline API to PTX Mapping

Here's a complete mapping of Pipeline APIs to their underlying PTX instructions:

| Pipeline API | Barrier Type | PTX Instruction |
|-------------|--------------|-----------------|
| `producer_acquire` (wait) | EmptyBarrier | `mbarrier.try_wait.parity` (spin) |
| `producer_acquire` (leader) | FullBarrier | `mbarrier.arrive.expect_tx` |
| `producer_commit` | FullBarrier | `mbarrier.complete_tx` (TMA auto) |
| `consumer_try_wait` | FullBarrier | `mbarrier.try_wait.parity` (once) |
| `consumer_wait` | FullBarrier | `mbarrier.try_wait.parity` (spin) |
| `consumer_release` | EmptyBarrier | `mbarrier.arrive` |

## 5. Producer APIs

### 5.1 producer_acquire

Waits for buffer space and sets expected transaction bytes:

```cpp
// Source: include/cutlass/pipeline/sm90_pipeline.hpp:511-528
CUTLASS_DEVICE
void producer_acquire(uint32_t stage, uint32_t phase) {
  // Step 1: Wait for consumer to release the buffer
  empty_barrier_ptr_[stage].wait(phase);

  // Step 2: Leader thread sets expected transaction bytes
  if (params_.is_leader) {
    full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
  }
}
```

**What happens:**
1. All producer threads wait on `EmptyBarrier` until the buffer is available
2. Only the **leader thread** (one per warp) calls `arrive_and_expect_tx` to set up the `FullBarrier` with expected bytes

### 5.2 producer_get_barrier

Returns the barrier pointer for TMA to signal completion:

```cpp
// Source: include/cutlass/pipeline/sm90_pipeline.hpp:456-459
CUTLASS_DEVICE
ProducerBarrierType* producer_get_barrier(PipelineState state) {
  return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[state.index()]);
}
```

This pointer is passed to TMA operations so the hardware can automatically signal the barrier when data transfer completes.

### 5.3 producer_tail

Prevents premature producer exit in cluster scenarios:

```cpp
// Source: include/cutlass/pipeline/sm90_pipeline.hpp:447-454
CUTLASS_DEVICE
void producer_tail(PipelineState state) {
  for (int count = 0; count < Stages; ++count) {
    empty_barrier_ptr_[state.index()].wait(state.phase());
    ++state;
  }
}
```

This ensures all consumers have finished using all pipeline stages before the producer block exits.

## 6. Consumer APIs

### 6.1 consumer_try_wait

Non-blocking check if data is ready:

```cpp
// Source: include/cutlass/pipeline/sm90_pipeline.hpp:590-597
CUTLASS_DEVICE
ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
  if (skip_wait) {
    return {BarrierStatus::WaitDone};
  }
  bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
  return {static_cast<BarrierStatus>(barrier_status)};
}
```

Returns:
- `BarrierStatus::WaitDone` (1): Data is ready
- `BarrierStatus::WaitAgain` (0): Data not ready yet

### 6.2 consumer_wait

Blocking wait for data:

```cpp
// Source: include/cutlass/pipeline/sm90_pipeline.hpp:611-620
CUTLASS_DEVICE
void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
  if (barrier_token != BarrierStatus::WaitDone) {
    full_barrier_ptr_[stage].wait(phase);
  }
  // If already WaitDone, skip the wait
}
```

### 6.3 consumer_release

Signals that the buffer can be reused:

```cpp
// Source: include/cutlass/pipeline/sm90_pipeline.hpp:628-630
CUTLASS_DEVICE
void consumer_release(uint32_t stage, uint32_t skip = false) {
  empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip));
}
```

This decrements the `EmptyBarrier` arrival count, allowing the producer to reuse the buffer.

## 7. mbarrier 64-bit Structure

The mbarrier is a 64-bit hardware structure in shared memory:

| Field | Bits | Description |
|-------|------|-------------|
| Phase Bit | 1 | Toggles on barrier completion |
| Pending TX Count | ~20 | Expected bytes (for TransactionBarrier) |
| Arrival Count | ~20 | Remaining arrivals needed |

**Completion Condition:**
```
Pending TX Count == 0 AND Arrival Count == 0 → Phase Bit flips
```

## 8. Complete Workflow Example

```cpp
// Producer Thread (TMA loader)
PipelineState producer_state = make_producer_start_state<Pipeline>();

for (int k = 0; k < num_tiles; ++k) {
  // 1. Acquire buffer (wait for space + set expected bytes)
  pipeline.producer_acquire(producer_state);

  // 2. Get barrier for TMA
  auto* barrier = pipeline.producer_get_barrier(producer_state);

  // 3. Issue TMA load (hardware will auto-complete barrier)
  copy(tma_load, gmem_tensor, smem_tensor, barrier);

  ++producer_state;
}

// 4. Wait for all consumers before exit
pipeline.producer_tail(producer_state);


// Consumer Thread (MMA compute)
PipelineState consumer_state{0, 0, 0};

for (int k = 0; k < num_tiles; ++k) {
  // 1. Try wait (opportunistic)
  auto token = pipeline.consumer_try_wait(consumer_state);

  // 2. Do some other work...

  // 3. Finalize wait if needed
  pipeline.consumer_wait(consumer_state, token);

  // 4. Use data in MMA operations
  gemm(smem_tensor, accumulators);

  // 5. Release buffer for producer
  pipeline.consumer_release(consumer_state);

  ++consumer_state;
}
```

## 9. Key Takeaways

1. **SM90 Specific**: This pipeline mechanism is designed for NVIDIA Hopper architecture (SM90+)

2. **Hardware Acceleration**: TMA automatically signals barriers on completion - no software intervention needed

3. **Phase-Based Synchronization**: The phase bit enables reuse of barriers across iterations

4. **Cluster Support**: Barriers can operate across CTAs in a cluster via `mapa` address remapping

5. **Transaction Barriers**: `ClusterTransactionBarrier` tracks byte counts for TMA operations, while `ClusterBarrier` uses simple arrival counting

## References

- [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass)
- [NVIDIA PTX ISA - mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- [CUDA Programming Guide - Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-barrier)
