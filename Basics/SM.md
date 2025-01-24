# SM

In this lesson we will break down the key components of Nvidia Hopper GPU architecture, focusing on SMs (Streaming Multiprocessors), L2 cache sections, and memory controllers.

Streaming Multiprocessors (SMs) in Nvidia Hopper:

Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the cores of CPUs. That is, SMs both execute computations and store state available for computation in registers, with associated caches. Compared to CPU cores, GPU SMs are simple, weak processors. Execution in SMs is pipelined within an instruction (as in almost all CPUs since the 1990s) but there is no speculative execution or instruction pointer prediction (unlike all contemporary high-performance computation CPUs).

SMs are the building blocks that enable massive parallelism, allowing GPUs to handle thousands of threads simultaneously. Each SM operates independently, managing threads, memory, and computations within its domain.

Now there are a lot of things going under the hood in SMs. To understand that we need to understand what is inside SM. So here is a basic overview of the components of SM - 

### Components of SM

#### **1. CUDA Cores**  
##### **Technical Details**  
- **Architecture**: CUDA cores are simplified arithmetic logic units (ALUs) optimized for parallel floating-point and integer operations. Unlike CPU cores, which handle complex instruction pipelines and branch prediction, CUDA cores focus on throughput over latency.  
- **Precision Modes**:  
  - **FP32**: Primary focus for graphics and general-purpose compute (e.g., matrix operations in AI).  
  - **FP64**: Double-precision cores for scientific simulations (fewer in consumer GPUs, more in data-center GPUs like A100).  
  - **INT32/INT4**: Specialized cores for integer operations (e.g., AI quantization, ray traversal).  
- **SM Organization**:  
  - In **Ampere** (e.g., GA102 in RTX 3090), each SM contains **128 FP32 CUDA cores** and **64 INT32 cores**, allowing concurrent FP32+INT32 execution.  
  - **Hopper** (H100) introduces **FP8 Tensor Cores**, emphasizing AI workloads.  

##### **Practical Implications**  
- **Throughput vs. Latency**: GPUs prioritize executing many threads in parallel rather than speeding up single threads.  
- **Data Type Selection**: Using lower precision (FP16/INT8) can double throughput on architectures with dedicated cores (e.g., Tensor Cores).  

---

#### **2. Warp Schedulers**  
##### **Technical Details**  
- **Warp Mechanics**:  
  - A warp (32 threads) is the smallest executable unit. Threads in a warp follow the same instruction path (SIMT).  
  - **Warp Divergence**: If threads branch (e.g., `if/else`), both paths execute sequentially, halving efficiency. Use `__syncwarp()` or restructuring (e.g., pre-sorting data) to mitigate.  
- **Scheduler Types**:  
  - **GigaThread Engine**: Global scheduler distributing blocks to SMs.  
  - **Per-SM Schedulers**:  
    - **Volta+**: 4 warp schedulers/SM, enabling 2 instructions per clock (via **independent thread scheduling**).  
    - **Ampere**: Enhanced to handle concurrent FP32 and INT32 operations.  

##### **Latency Hiding**  
- **Memory Latency**: When a warp stalls (e.g., global memory access), schedulers switch to ready warps. High **occupancy** (active warps/SM) ensures better latency hiding.  
- **Occupancy Calculator**: NVIDIA provides tools to estimate occupancy based on thread block size and resource usage (registers/shared memory).  

---

#### **3. Registers**  
##### **Technical Details**  
- **Per-Thread Storage**: Each thread has a private register set (e.g., 255 registers/thread on Ampere).  
- **Register Spilling**: If registers exceed hardware limits, data spills to slower memory (local/global), crippling performance. Use `--ptxas-options=-v` to check spills.  
- **Occupancy Trade-off**:  
  - **Example**: An SM with 65,536 registers can host:  
    - 2048 threads if each uses 32 registers (2048 × 32 = 65,536).  
    - 1024 threads if each uses 64 registers (1024 × 64 = 65,536).  

#### **4. Shared Memory**  
**Definition**:  
Shared Memory in GPU architectures is a high-speed, programmable memory space located on the Streaming Multiprocessor (SM) itself. It is explicitly managed by the programmer and shared among all threads within a single thread block. Unlike global memory (off-chip DRAM), shared memory provides **nanosecond-level latency** (similar to L1 cache) due to its on-chip location, making it orders of magnitude faster than global memory. Its primary purpose is to enable efficient data sharing and collaboration between threads in a block, reducing redundant global memory accesses.


##### **Function**:  
1. **Data Reuse and Collaboration**:  
   - **Stencil Operations**: In algorithms like image convolution or PDE solvers, threads often require neighboring data (e.g., a 3x3 pixel grid). Shared memory allows threads to load a block of data once, reuse it across multiple threads, and avoid repeated global memory fetches.  
   - **Reductions**: Operations like sum, max, or min across a thread block benefit from shared memory. Threads compute partial results, store them in shared memory, and iteratively combine results in parallel (e.g., using a tree-based reduction).  
   - **Matrix Multiplication**: When multiplying matrices, threads load sub-matrices (tiles) into shared memory to exploit spatial locality and minimize global memory bandwidth usage.

2. **Explicit Management**:  
   - Programmers must declare the size of shared memory statically (e.g., `__shared__ float tile[32][32];`) or dynamically (via kernel configuration).  
   - **Synchronization**: Threads within a block must synchronize using `__syncthreads()` to ensure all data is written to shared memory before others read it.  
   - **Bank Conflict Avoidance**: Shared memory is divided into 32 banks (on most GPUs). Concurrent accesses to the same bank cause serialization. Programmers use techniques like memory padding or strided access patterns to mitigate conflicts.


##### **Capacity and Architecture Dependence**:  
- **Size**: Typically 64–128 KB per SM, but this varies by architecture (e.g., NVIDIA Ampere GPUs allocate up to 164 KB).  
- **Configurability**: On architectures like Fermi or Kepler, shared memory and L1 cache partition a 64 KB memory pool. Programmers can prioritize shared memory (e.g., 48 KB shared + 16 KB L1) using `cudaFuncSetCacheConfig()`.  
- **Limitations**: Overuse can lead to resource contention, limiting the number of active thread blocks per SM.

---

#### **5. L1 Cache/Texture Units**  
##### **L1 Cache**:  
**Function**:  
- The L1 cache is a hardware-managed memory that automatically caches frequently accessed data from global or local memory. It reduces latency by storing recently used data closer to the SM.  
- **Spatial and Temporal Locality**: Optimized for access patterns where data is reused (e.g., looping over an array multiple times).  

**Shared Use with Shared Memory**:  
- In older architectures (e.g., Fermi), L1 and shared memory are partitioned from a unified 64 KB memory pool. Newer architectures (Volta+) decouple them, allowing simultaneous allocation.  
- **Trade-offs**: Increasing shared memory size reduces L1 cache, which may impact caching efficiency for irregular memory access patterns.


##### **Texture Units**:  
**Function**:  
- **Hardware-Accelerated Filtering**: Texture units are specialized for graphics tasks like bilinear/trilinear filtering, which interpolate texel values (e.g., smoothing pixels in zoomed images).  
- **Cache Optimization**: Texture memory uses a dedicated cache optimized for 2D/3D spatial locality, making it ideal for image processing (e.g., volume rendering, computer vision).  
- **Non-Graphics Use Cases**:  
  - Read-only data with spatial access patterns (e.g., medical imaging).  
  - Boundary handling via automatic clamping/wrapping of out-of-bounds coordinates.  

**Programming Interface**:  
- Accessed via CUDA’s texture API (e.g., `texture<float, 2> texRef;`).  
- Supports normalized coordinates and multiple data formats (e.g., `cudaFilterModeLinear` for interpolation).

---

#### **6. Specialized Cores**  
##### **Tensor Cores** (Volta+ Architectures):  
**Function**:  
- **Mixed-Precision Matrix Operations**: Accelerate matrix multiply-accumulate (MMA) operations, such as \( D = A \times B + C \), where \( A, B, C, D \) can be FP16, BF16, INT8, or FP64 matrices.  
- **Throughput**: A single Tensor Core can compute a 4x4x4 matrix multiplication per clock cycle, delivering up to 125 TFLOPS (FP16) on NVIDIA A100 GPUs.  

**Use Cases**:  
- **Deep Learning**: Training/inference with frameworks like TensorFlow/PyTorch, where large matrix multiplications dominate (e.g., fully connected layers in neural networks).  
- **High-Performance Computing (HPC)**: Solving linear algebra problems (e.g., LU decomposition).  

**Programming**:  
- Accessed via CUDA’s WMMA (Warp Matrix Multiply-Accumulate) API or libraries like cuBLAS/cuDNN.  
- Requires data to be formatted in specific layouts (e.g., 16x16 FP16 tiles for MMA operations).  

---

##### **Performance Impact**:  
- **Tensor Cores**: Reduce training times for AI models from weeks to days. For example, ResNet-50 training can be accelerated by 6x using Tensor Cores.  
- **RT Cores**: Enable real-time ray tracing at 60+ FPS in games like *Cyberpunk 2077*, which would otherwise require orders of magnitude more computation on CUDA cores.  

---

##### **Architectural Considerations**  
- **Shared Memory vs L1 Cache**: Choose shared memory for predictable, collaborative data reuse; rely on L1 for irregular access with locality.  
- **Tensor/RT Core Availability**: Requires GPU architectures from Volta (2017) or Turing (2018) onward.  
- **Trade-offs**: Specialized cores increase silicon area but provide unmatched efficiency for targeted workloads.

---

#### **7. Warps**  
**Definition**:  
A **warp** is the fundamental unit of execution in a GPU, consisting of a group of **32 threads** that operate in **lockstep** (SIMD: Single Instruction, Multiple Data). Warps are managed by the Streaming Multiprocessor (SM) and represent the smallest schedulable unit of work. All threads in a warp execute the same instruction simultaneously, leveraging parallelism at the instruction level.

---

##### **Function**:  
1. **Scheduling and Execution**:  
   - The SM schedules warps, not individual threads. At every clock cycle, the **warp scheduler** selects a warp that is ready to execute (e.g., not stalled waiting for data).  
   - **Lockstep Execution**: All 32 threads in a warp execute the same instruction on different data. For example, in a vector addition kernel, a warp might compute 32 elements of the vector concurrently.  
   - **Efficiency**: Warps enable massive parallelism by allowing the SM to hide latency through rapid context switching between warps (see [Latency Hiding](#9-latency-hiding)).  

2. **Divergence Handling**:  
   - **Branch Divergence**: If threads within a warp follow different execution paths (e.g., `if-else` statements), the warp **serializes** execution. For example, threads taking the `if` branch execute first, while others idle, followed by those taking the `else` branch.  
   - **Performance Penalty**: Divergence can drastically reduce throughput. A warp with divergent branches may require 2–32x more cycles to complete.  
   - **Mitigation Strategies**:  
     - Avoid branch conditions that vary within a warp (e.g., use warp-wide conditions).  
     - Restructure code to group threads with similar execution paths (e.g., using `__ballot_sync()` to coordinate branches).  

---

#### **8. Memory Hierarchy**  
The GPU memory hierarchy is a tiered structure designed to balance speed, capacity, and programmability:  

1. **Registers** (Fastest):  
   - **Per-Thread Storage**: Each thread has dedicated registers (e.g., 255 registers/thread on NVIDIA Ampere).  
   - **Zero Latency**: Register access is the fastest, as values are stored directly in the SM’s register file.  
   - **Limitations**: Excessive register usage reduces the number of concurrent threads (occupancy).  

2. **Shared Memory**:  
   - **Block-Level Shared Storage**: On-chip memory accessible to all threads in a thread block (see [Shared Memory](#4-shared-memory)).  
   - **Use Cases**: Storing reusable data (e.g., matrix tiles in GEMM kernels) or partial results (e.g., reductions).  

3. **L1/Texture Cache**:  
   - **L1 Cache**: Caches global/local memory accesses with spatial/temporal locality. Automatically managed by hardware.  
   - **Texture Cache**: Optimized for 2D/3D spatial locality, supporting hardware-accelerated filtering (e.g., bilinear interpolation).  

4. **Global Memory** (Slowest):  
   - **GPU DRAM**: High-capacity memory (e.g., 24 GB on NVIDIA RTX 4090) but high latency (~400 cycles).  
   - **Optimization**:  
     - **Coalesced Access**: Combine memory requests from adjacent threads into a single transaction (e.g., access contiguous 128-byte aligned blocks).  
     - **Memory Compression**: Newer architectures (e.g., Ampere) use delta color compression to reduce bandwidth usage.  

---

#### **9. Latency Hiding**  
**Mechanism**:  
- **Warp-Level Multithreading**: When a warp stalls (e.g., waiting for global memory), the SM immediately switches to another **ready warp** (one with all operands available).  
- **Occupancy**: The ratio of active warps to the maximum supported by the SM (e.g., 64 warps/SM on Ampere). Higher occupancy improves latency hiding.  

**Key Concepts**:  
1. **Memory-Computation Overlap**:  
   - While Warp A waits for data, Warp B executes arithmetic operations.  
   - Requires sufficient independent work (warps) to keep the SM busy.  
2. **Limitations**:  
   - **Resource Constraints**: Limited registers/shared memory per SM restrict the number of active warps.  
   - **Kernel Design**: Poorly structured kernels (e.g., excessive synchronization) reduce warp schedulability.  

**Example**:  
In matrix multiplication, while one warp loads the next tile from global memory, another warp computes the dot product for the current tile.  

---

#### **10. Thread Block Execution**  
**Assignment**:  
- Each thread block is statically assigned to an SM at launch and remains there until completion.  
- **SM Resources**: The number of blocks per SM depends on:  
  - **Register Usage**: Each thread consumes registers (e.g., 64 threads/block × 32 registers/thread = 2,048 registers/block).  
  - **Shared Memory**: Blocks declare fixed shared memory (e.g., 48 KB/block).  
  - **Thread Slots**: SMs have a maximum thread capacity (e.g., 2,048 threads/SM on Ampere).  

**Resource Limits**:  
- **Partitioning**: If a block requires 64 KB of shared memory and the SM has 128 KB, only two blocks can reside on the SM.  
- **Occupancy Calculator**: Tools like NVIDIA’s `CUDA_Occupancy_Calculator.xls` help optimize block size for maximum occupancy.  

**Optimization Strategies**:  
- **Block Size**: Use multiples of warp size (32 threads) to avoid underutilized warps.  
- **Dynamic Partitioning**: Adjust shared memory/register usage to fit more blocks/SM.  

---

#### **Architectural Considerations**  
- **Warp Efficiency**: Minimize divergence and maximize instruction-level parallelism (ILP) to keep warps active.  
- **Memory Hierarchy**: Prioritize register/local memory for thread-private data, shared memory for collaboration, and optimize global memory access patterns.  
- **Latency Hiding**: Design kernels to maximize independent work (warps) and balance compute/memory operations.  
- **Block Configuration**: Tailor block size to SM resource limits for optimal occupancy.  


---

Now different Architectures have different specs. So to cut the hassle here's a basic overview - 

### **Comparison of All the GPU Architectures**  

| Architecture | CUDA Cores/SM          | Warp Schedulers/SM | Tensor Cores | RT Cores | Memory Tech      | Key Features                                      | Process Node     |  
|--------------|------------------------|--------------------|--------------|----------|------------------|---------------------------------------------------|------------------|  
| **Fermi**    | 32 FP32                | 2                  | N/A          | N/A      | GDDR5            | First ECC memory, shared memory/L1 cache          | 40/28nm          |  
| **Kepler**   | 192 FP32               | 4                  | N/A          | N/A      | GDDR5            | Dynamic Parallelism, Hyper-Q, 64 DP units         | 28nm             |  
| **Maxwell**  | 128 FP32               | 4                  | N/A          | N/A      | GDDR5            | Unified shared memory/L1 cache, 2x perf/W         | 28nm             |  
| **Pascal**   | 64 FP32                | 2                  | N/A          | N/A      | HBM2/GDDR5X      | Unified memory, NVLink 1.0, 16nm FinFET           | 16nm             |  
| **Volta**    | 64 FP32 + 64 INT32     | 4                  | 1st-gen      | N/A      | HBM2             | Independent Thread Scheduling, Tensor Cores       | 12nm             |  
| **Turing**   | 64 FP32 + 64 INT32     | 4                  | 2nd-gen      | 1st-gen  | GDDR6            | RT Cores, concurrent FP/INT, INT8/INT4 Tensor Cores | 12nm             |  
| **Ampere**   | 64 FP32 + 64 INT32     | 4                  | 3rd-gen      | 2nd-gen  | GDDR6X/HBM2      | Structural sparsity, 2x AI throughput, PCIe Gen4  | 7nm              |  
| **Hopper**   | 128 FP32 + 64 INT32    | 4                  | 4th-gen      | N/A      | HBM3             | FP8 Tensor Cores, Thread Block Clusters, NVLink 4.0 | TSMC 4N          |  
| **Blackwell**| 128 unified FP32/INT32 | 4                  | 5th-gen      | 4th-gen  | GDDR7/HBM3e      | FP4 Tensor Cores, AI Management Processor, 10 TB/s interconnect | TSMC 4NP         |  


