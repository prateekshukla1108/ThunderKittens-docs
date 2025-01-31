
## Trying out your kernel 

We can run a kernel from c++ or from Python. 

Running from C++:

* We need a Makefile like: [Makefile](https://github.com/HazyResearch/ThunderKittens/blob/tk_gen/simple_kernels/micro_add/Makefile)   
* We need a main function just like any code as specified here: [Main function in cpp](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/harness.impl#L23)  
* We run some set up code to create dummy tensors on the GPU  
  * Save to file: [Save tensor values to file](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/gentests.py#L31)   
  * And load them in from the file: [Load tensor values from file](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/harness.impl#L31)   
* Then we dispatch our kernel: [Calling dispatch from micro.cu](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/harness.impl#L48)    
* We add this harness.impl file to our CUDA kernel file to make sure that this main function gets called: [Add the harness](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/micro.cu#L61)  
* This code checks whether the kernel’s result matches the pytorch result: [Check code](https://github.com/HazyResearch/ThunderKittens/blob/87b30649818d93ecae61827ef4470545cfd85cc1/simple_kernels/micro_add/harness.impl#L53) 

Example run sequence:

```c
python gentests.py randn
make clean && make 
./micro randn.txt
```

   
Running from Python: 

* This code shows how to set up to run from python: [Run from python](https://github.com/HazyResearch/ThunderKittens/tree/tk_gen/simple_kernels/bind_add) 

## 

## TK Kernel Examples \- Basic {#tk-kernel-examples---basic}

Walk through these three kernels to develop a “basic” TK understanding.

1. Simple kernels: [folder](https://github.com/HazyResearch/ThunderKittens/tree/tk_gen/simple_kernels)  
2. Layer norm: [folder](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/layernorm/non_pc)  
3. FFT convolution: [folder](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/fftconv/non_pc)   
4. Attention: [https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/attn/demo/4090.cu](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/attn/demo/4090.cu)   


Let's now dive deeper into how things work under the hood and how ThunderKittens's kernel template helps us achieve peak performance
