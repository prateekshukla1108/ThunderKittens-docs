## Adding two tensors
Let's try to impliment a basic TK kernel to add two tensors - 

* Here is the kernel to add two tensors - 

```c
#include "kittens.cuh"
using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 16
#define _col 32

struct micro_globals {
    using _gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    _gl x, o;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _col> (&o_s) = al.allocate<st_fl<_row, _col>>();

    // register memory 
    rt_fl<_row, _col> x_reg_fl;

    // load from HBM to shared
    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg_fl, x_s);
    __syncthreads();

    // x (dst) = x (src b) + x (src a)
    add(x_reg_fl, x_reg_fl, x_reg_fl);
    __syncthreads();

    // store from register to shared
    store(o_s, x_reg_fl);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}

void dispatch_micro( float *d_x, float *d_o ) {
    using _gl = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    using globals = micro_globals;
    _gl  x_arg{d_x, 1, 1, _row, _col};
    _gl  o_arg{d_o, 1, 1, _row, _col};
    globals g{x_arg, o_arg};
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<1,32,mem_size>>>(g);
    cudaDeviceSynchronize();
}
#include "harness.impl"

```

* This is what it does, when written in Python: [Python](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/gentests.py#L24) 

### **Key Components**
1. **Header and Namespace**:
   - `#include "kittens.cuh"`: Includes the `kittens` library, which provides abstractions for CUDA operations.
   - `using namespace kittens`: Uses the `kittens` namespace for convenience.

2. **Constants**:
   - `NUM_THREADS`: Defined as `kittens::WARP_THREADS` (typically 32 threads, the size of a CUDA warp).
   - `_row` and `_col`: Define the dimensions of the tensor (16x32, matching the Python script's `N` and `D`).

3. **Global Variables**:
   - `micro_globals`: A struct that holds global memory pointers for input (`x`) and output (`o`) tensors. These are represented as `gl` (global layout) objects with specific dimensions.

4. **CUDA Kernel**:
   - `micro_tk`: The CUDA kernel that performs the computation.
     - **Shared Memory**: Allocates shared memory for intermediate storage of tensors (`x_s` and `o_s`).
     - **Register Memory**: Uses register memory (`x_reg_fl`) for computations.
     - **Memory Workflow**:
       1. Loads data from global memory (`g.x`) to shared memory (`x_s`).
       2. Loads data from shared memory (`x_s`) to register memory (`x_reg_fl`).
       3. Performs the addition operation (`add(x_reg_fl, x_reg_fl, x_reg_fl)`), equivalent to `x + x` in the Python script.
       4. Stores the result from register memory (`x_reg_fl`) back to shared memory (`o_s`).
       5. Stores the result from shared memory (`o_s`) back to global memory (`g.o`).

5. **Kernel Launch**:
   - `dispatch_micro`: Prepares and launches the CUDA kernel.
     - Initializes global memory pointers (`x_arg` and `o_arg`) for input and output tensors.
     - Sets the shared memory size for the kernel.
     - Launches the kernel with 1 block and 32 threads (1 warp).
     - Synchronizes the device to ensure the kernel completes.

6. **Harness**:
   - `#include "harness.impl"`: Likely includes additional boilerplate or utility code for running the kernel.

---

### **Correspondence to Python Script**
| **Python/PyTorch**                     | **CUDA Code**                                                                 |
|----------------------------------------|-------------------------------------------------------------------------------|
| `x = torch.ones((B, N, D), ...)`       | `x_arg{d_x, 1, 1, _row, _col}` (initializes input tensor in global memory).   |
| `o = x + x`                            | `add(x_reg_fl, x_reg_fl, x_reg_fl)` (performs addition in register memory).   |
| `o.to(torch.float32).flatten()`        | `store(g.o, o_s)` (stores result back to global memory).                      |
| Saving to file (`{TESTNAME}.txt`)      | Not explicitly done in CUDA code (handled by host code or harness).           |

---

### **Memory Hierarchy in CUDA**
Your CUDA code explicitly manages the memory hierarchy:
1. **Global Memory (HBM)**: Input (`d_x`) and output (`d_o`) tensors reside here.
2. **Shared Memory**: Used for intermediate storage (`x_s` and `o_s`).
3. **Register Memory**: Used for computations (`x_reg_fl`).

This explicit management is necessary for performance optimization in CUDA but is abstracted away in PyTorch.

---

### **Performance Considerations**
- **Shared Memory**: The use of shared memory (`__shared__`) reduces global memory access, which is slower.
- **Register Memory**: Performing computations in registers is faster than using shared or global memory.
- **Thread Synchronization**: `__syncthreads()` ensures proper synchronization between threads when accessing shared memory.

---

### **How It Works**
1. The host code (`dispatch_micro`) initializes the input and output tensors in global memory.
2. The CUDA kernel (`micro_tk`) is launched with 1 block and 32 threads (1 warp).
3. The kernel:
   - Loads data from global memory to shared memory.
   - Transfers data from shared memory to registers.
   - Performs the addition in registers.
   - Stores the result back to shared memory and then to global memory.
4. The host code synchronizes the device to ensure the kernel completes.

