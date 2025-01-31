
## TK Basic Overview

1. **Learning about TK**

TK provides a set of *templated wrappers* over raw CUDA in C++ to *uplevel the programming experience* and make it more PyTorch-like. TK functions call CUDA under the hood, but make it so that you don’t have to worry about certain aspects of CUDA when you program. 

Take a look at this example for adding two tensors:

* Here is potentially the most basic kernel: [TK Kernel](https://github.com/HazyResearch/ThunderKittens/blob/tk_gen/simple_kernels/micro_add/micro.cu)   
* This is what it does, when written in Python: [Python](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/gentests.py#L24) 

Understanding how tensors from Pytorch are represented in TK. As **tiles**. 

* Each tile has a shape (height H, width W) and a data type (float, bf16, fp8, half)  
* We can create a tile in register or shared memory  
  * [Shared example](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L17)   
  * [Register example](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L23) 

Loading and storing data from HBM:

* We will need to load data from our tensors in Pytorch/HBM of shape {B, H, N, D} into these tiles. We index into which part of the tensor we want to load using {b, h, n, d} style tuples as shown in the following example:  
  * [Load HBM to Shared example](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L25)     
  * [Store shared to HBM](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L42)   
* Some examples:  
  * If I have a tensor of shape \[8, 16, 1024, 64\], like \[batch, heads, sequence length, head dimension\] in machine learning, and I am loading 64x64 tiles, I might:  
    * Parallelize over the batches and heads when I launch my grid so  
      * Head\_id \= blockIidx.x  
      * Batch\_id \= blockIdx.y  
      * And then in my kernel syntax, to load the 3rd chunk along the sequence length, I will load {batch\_id, head\_id, 2, 0} with 0-indexing  
  * An index should also just be 0 if it’s not present/relevant to your PyTorch tensor

Operating over tiles in shared or register memory. Think of these functions like PyTorch functions (torch.sum(), torch.matmul()..) except for now it’s kittens functions. 

* Each function takes inputs (dst tile, src a tile, src b tile)  
* [Add function example](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L34)   
* The suite of functions that can be run over tiles includes:  
  * Registers tiles specified here: [folder](https://github.com/HazyResearch/ThunderKittens/tree/main/include/ops/warp/register/tile)  
  * Shared tiles specified here: [folder](https://github.com/HazyResearch/ThunderKittens/tree/main/include/ops/warp/shared/tile) 

Setting up the code to launch a kernel:

* [Global struct](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L8)   
  * This defines the *types* of the tensors you’ll be loading from PyTorch.   
  * We can see a data type: floating point (fp32)  
  * We can see four numbers that can be {-1} or {c}, for some positive number c. These specify the shape of the incoming tensor as {Batch, Heads, Height, Width} so the max tensor we support is 4D. If you have a 2D tensor for instance, then just set Batch and Heads to 1\.  
    * \-1 means that the dimension is specified at runtime  
    * C means that the dimension is specified at compile time  
  * And then we specify a tile (e.g., st\_fl\<\_row, \_col\>) which shows the size of the tiles we’ll be loading in from the HBM tensor at any given time.  
* [Dispatch function](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L46)   
  * We need to instantiate our global struct and specify all the inputs we’re passing into the kernel as different global objects: [Example](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L47)  
  * Set up the kernel: [Setup](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L53)  
  * Launch the kernel with some grid: [Example](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L58)   


With that let's now discuss how you can try out your own kernel
