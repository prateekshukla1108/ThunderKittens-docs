## Advanced Overview

For this section, I’ll talk about TK’s *kernel template* which helps to get peak performance. We will specifically focus on understanding this matrix multiply kernel: [kernel](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/matmul/H100/matmul.cu)

* Template creation. Instantiated as shown: [template](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L18).   
* Logic. The template differentiates two types of workers (i.e., sets of warps/warpgroups) – producers and consumers. The idea is for the producers to focus on loading/storing data between shared memory and HBM. The consumers focus on running computations on data in shared memory and registers. Overall, the template helps manage **asynchronous execution** amongst these two types of workers, which is crucial to peak performance kernels.   
  * [Producer workers](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L49), who do functions in this block. The functions include:  
    * Setup: any producer specific setup you need.   
      * Generally we will always call this line to decrease registers when running an H100 kernel: [register re-alloc](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L51). This is a nice feature to be able to help the compute workers use larger tiles. The producers don’t really need their registers since they’re busy wrangling global/shared memory :)    
    * Load: load data from HBM to shared memory  
    * Store: stores data from shared memory to HBM  
  * [Consumer workers](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L65), who do functions in this block. The functions include:   
    * Setup: any consumer specific setup you need  
      * Corollary to the producers; steal more registers from the register pool [register realloc](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L67) :)    
    * Compute:   
      * Generally this is your core algorithmic logic  
    * Finish (optional)   
      * Any tasks that the consumers need to perform before they shut down. E.g., sometimes they can flush data to HBM instead of the producers, like in this matmul kernel: [flush accum to HBM](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L79)   
  * [Common setup](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L28) between producers and consumers  
* Memory block rules: depending on who needs to access the memory tiles, define it in the appropriate block.   
  * Globals are like we’re familiar with from the above sections: [globals](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L11)   
  * Input block is shared memory across producer and consumer workers: [input](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L12)  
  * Finish block defines shared memory used to write outputs out to HBM from the finish function: [finish](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L13)    
  * Common state holds values that are shared between producer, consumer workers throughput the execution flow: [common state](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L14)  
  * Consumer state holds register values typically that are required for the consumer to carry out its computation in the compute function: [consumer state](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/matmul/H100/matmul.cu#L15)  
  * Scratch block: [scratch](https://github.com/HazyResearch/ThunderKittens/blob/99ecff7f69d8ce96dcbad431079c67018d9c9d6b/kernels/torch_scaled/scaled_matmul.cu#L39)   
* When do different functions in the template get called and in what order?  
  * The details are here [template backend](https://github.com/HazyResearch/ThunderKittens/blob/main/prototype/lcsf/lcsf.cuh) however we hope that you *do not* need to worry about the backend when using the template

Details

* The TK template uses persistent kernels

FAQ: 

1. What to think about when writing a kernel?   
   1. L2 reuse: are there values that need to be used by multiple thread blocks? Then perhaps instead of each thread block loading individually from HBM, they can pull from the L2 cache *if we setup the load access patterns in the correct orders*  
   2. Occupancy:   
      1. How many different workers are executing at the same time? Number of producers/consumers? What complementary work are they doing?  
      2. How does that influence the tile sizes that we can use?   
      3. We want tiles to be big enough to use bank-conflict free swizzle modes  
   3. Tensor cores: is there a way to restructure the computation/algorithm to use matrix multiplications  
2. Should I familiarize myself with template (vs just write TK code with out it)?   
   1. For an H100 kernel, where async compute is very important, probably/yes

## 

## TK Kernel Examples 

Walk through these three kernels to develop an “advanced” TK understanding (i.e., uses the template discussed in the prior section).

* The matmul is a great place to start, which we walked through in the prior section\!  
* Here you can see how the [basic FFT convolution kernel](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/fftconv/non_pc) gets mapped to a [templated FFT convolution kernel](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/fftconv/pc)  
* [Mamba-2](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/mamba2) uses the TK template  
* [Rotary](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/rotary) uses the TK template   
