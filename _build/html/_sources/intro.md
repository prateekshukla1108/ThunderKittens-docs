# ThunderKittens

### Tile primitives for speedy kernels

<div align="center" >
    <img src="logo.png" height=350 alt="ThunderKittens logo" style="margin-bottom:px"/> 
</div>

<br>
<br>

ThunderKittens is a framework to make it easy to write fast deep learning kernels in CUDA (and, soon, MPS, and eventually ROCm and others, too!)

ThunderKittens is built around three key principles:
1. Simplicity. ThunderKittens is stupidly simple to write.
2. Extensibility. ThunderKittens embeds itself natively, so that if you need more than ThunderKittens can offer, it won’t get in your way of building it yourself.
3. Speed. Kernels written in ThunderKittens should be at least as fast as those written from scratch -- especially because ThunderKittens can do things the “right” way under the hood. We think our Flash Attention 3 implementation speaks for this point.

<div align="center" >
    <img src="assets/attn.png" height=600 alt="Flash Attention 3, but with kittens!" style="margin-bottom:px"/> 
</div>


Join us on Discord to get involved: [ThunderKittens channel @ GPU Mode Discord](https://discord.com/channels/1189498204333543425/1300872762163728550)!!!! Here is the invite link to GPU mode: https://discord.gg/gpumode

ThunderKittens is built from the hardware up -- we do what the silicon tells us. And modern GPUs tell us that they want to work with fairly small tiles of data. A GPU is not really a 1000x1000 matrix multiply machine (even if it is often used as such); it’s a manycore processor where each core can efficiently run ~16x16 matrix multiplies. Consequently, ThunderKittens is built around manipulating tiles of data no smaller than 16x16 values.

ThunderKittens makes a few tricky things easy that enable high utilization on modern hardware.
1. Tensor cores. ThunderKittens can call fast tensor core functions, including asynchronous WGMMA calls on H100 GPUs.
2. Shared Memory. I got ninety-nine problems but a bank conflict ain’t one.
3. Loads and stores. Hide latencies with asynchronous copies and address generation with TMA.
4. Distributed Shared Memory. L2 is _so_ last year.
5. Worker overlapping. Use our Load-Store-Compute-Finish template to overlap work and I/O.



