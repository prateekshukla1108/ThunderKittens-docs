

## Installation

To use Thunderkittens, there's not all that much you need to do with TK itself. It's a header only library, so just clone the repo, and include kittens.cuh. Easy money.

### Library requirements

But ThunderKittens does use a bunch of modern stuff, so it has fairly aggressive requirements.
 - CUDA 12.3+. Anything after CUDA 12.1 will _probably_ work, but you'll likely end up with serialized wgmma pipelines on H100s due to a bug in those earlier versions of CUDA. We do our dev work on CUDA 12.6, because we want our kittens to play in the nicest, most modern environment possible.
 - (Extensive) C++20 use -- TK runs on concepts. If you get weird compilation errors, chances are your gcc is out of date.

```bash
sudo apt update
sudo apt install gcc-11 g++-11
```

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11
```


```
sudo apt update
sudo apt install clang-11
```

If you can't find nvcc, or you experience issues where your environment is pointing to the wrong CUDA version:


```bash
export CUDA_HOME=/usr/local/cuda-12.6/
export PATH=${CUDA_HOME}/bin:${PATH} 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

### Installing pre-existing kernels

We've provided a number of TK kernels in the `kernels/` folder! To use these with PyTorch bindings:

1. Set environment variables. 

To compile examples, run `source env.src` from the root directory before going into the examples directory. (Many of the examples use the `$THUNDERKITTENS_ROOT` environment variable to orient themselves and find the src directory.

2. Select the kernels you want to build in `configs.py` file

3. Install:


```bash
python setup.py install
```

Finally, thanks to Jordan Juravsky for putting together a quick doc on setting up a [kittens-compatible conda](https://github.com/HazyResearch/ThunderKittens/blob/main/docs/conda_setup.md).


