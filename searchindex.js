Search.setIndex({"alltitles": {"": [[0, "id1"], [4, "id1"]], "1. CUDA Cores": [[2, "cuda-cores"]], "1. Registers (Fastest)": [[2, "registers-fastest"]], "10. Thread Block Execution": [[2, "thread-block-execution"]], "2. Shared Memory": [[2, "id1"]], "2. Warp Schedulers": [[2, "warp-schedulers"]], "3. L1/Texture Cache": [[2, "l1-texture-cache"]], "3. Registers": [[2, "registers"]], "4. Global Memory (Slowest)": [[2, "global-memory-slowest"]], "4. Shared Memory": [[2, "shared-memory"]], "5. L1 Cache and Texture Units": [[2, "l1-cache-and-texture-units"]], "6. Specialized Cores": [[2, "specialized-cores"]], "7. Warps": [[2, "warps"]], "8. Memory Hierarchy": [[2, "memory-hierarchy"]], "9. Latency Hiding": [[2, "latency-hiding"]], "<<<>>>": [[1, "id1"]], "Advanced Overview": [[0, null]], "Architectural Best Practices": [[2, "architectural-best-practices"]], "Attention": [[5, "attention"]], "Based": [[5, "based"]], "Block": [[1, "block"]], "Calculating Global Thread ID": [[1, "calculating-global-thread-id"]], "Comparison of All the GPU Architectures": [[2, "comparison-of-all-the-gpu-architectures"]], "Demos": [[6, "demos"]], "General setup": [[5, "general-setup"]], "Getting started": [[6, null]], "Grid": [[1, "grid"]], "Helper Types and Functions": [[1, "helper-types-and-functions"]], "Installation": [[7, null]], "Installing pre-existing kernels": [[7, "installing-pre-existing-kernels"]], "Kernel": [[1, "kernel"]], "Key Components of an SM": [[2, "key-components-of-an-sm"]], "L1 Cache": [[2, "l1-cache"]], "Learn more and get involved!": [[6, "learn-more-and-get-involved"]], "Library requirements": [[7, "library-requirements"]], "LoLCATS": [[5, "lolcats"]], "Memory Management": [[1, "memory-management"]], "NVIDIA\u2019s Programming Model": [[6, "nvidias-programming-model"]], "Nuts and Bolts of the operations": [[1, null]], "Other Restrictions": [[6, "other-restrictions"]], "Putting It All Together": [[1, "putting-it-all-together"]], "SM": [[2, null]], "Scopes": [[6, "scopes"]], "Some theory on Memory": [[1, "some-theory-on-memory"]], "TK Basic Overview": [[3, null]], "TK Demos: play with kittens!": [[5, null]], "TK Kernel Examples": [[0, "tk-kernel-examples"]], "TK Kernel Examples - Basic {#tk-kernel-examples\u2014basic}": [[4, "tk-kernel-examples-basic-tk-kernel-examples-basic"]], "Tensor Cores": [[2, "tensor-cores"]], "Tests": [[6, "tests"]], "Texture Units": [[2, "texture-units"]], "Threads": [[1, "threads"]], "ThunderKittens": [[8, null]], "ThunderKittens Manual": [[6, "thunderkittens-manual"]], "Tile primitives for speedy kernels": [[8, "tile-primitives-for-speedy-kernels"]], "Trying out your kernel": [[4, null]], "Typing": [[6, "typing"]], "Understanding CUDA Thread Indexing": [[1, "understanding-cuda-thread-indexing"]], "Your Demos!": [[5, "your-demos"]], "cudaDeviceSynchronize()": [[1, "cudadevicesynchronize"]], "cudaMalloc": [[1, "cudamalloc"]], "cudaMemcpy": [[1, "cudamemcpy"]], "dim3": [[1, "dim3"]]}, "docnames": ["Advanced/Overview", "Basics/Basics of CUDA", "Basics/SM", "Basics/tk overview basic", "Basics/trying kernels and examples", "Installation and Setup/Demo", "Installation and Setup/Getting Started", "Installation and Setup/Installation", "intro"], "envversion": {"sphinx": 62, "sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9}, "filenames": ["Advanced/Overview.md", "Basics/Basics of CUDA.md", "Basics/SM.md", "Basics/tk overview basic.md", "Basics/trying kernels and examples.md", "Installation and Setup/Demo.md", "Installation and Setup/Getting Started.md", "Installation and Setup/Installation.md", "intro.md"], "indexentries": {}, "objects": {}, "objnames": {}, "objtypes": {}, "terms": {"": [0, 1, 2, 3, 4, 7, 8], "0": [1, 2, 3, 6], "08838834764f": 6, "1": [1, 3, 5, 6, 7], "10": 1, "100": [6, 7], "1000x1000": 8, "1024": 3, "11": 7, "12": 7, "125f": 6, "128": [2, 6], "12nm": 2, "155": 6, "16": [1, 3, 6], "164": 2, "16nm": 2, "16x16": [1, 2, 6, 8], "16x16x1": 1, "16x64": 6, "192": 2, "1d": 1, "1st": 2, "2": [0, 3, 5, 6], "20": 7, "2024": 6, "24": 2, "255": [1, 2], "256": [1, 6], "28nm": 2, "2d": [1, 2, 3], "2nd": 2, "2x": 2, "3": [1, 5, 6, 7, 8], "32": [2, 6], "3b": 5, "3d": [1, 2], "3rd": [2, 3], "4": 6, "40": 2, "400": 2, "4090": [2, 6], "44269504089": 6, "48": 2, "4d": 3, "4n": 2, "4np": 2, "4th": 2, "4x4": 2, "5": 5, "5th": 2, "6": 7, "64": [2, 3, 6], "64x64": 3, "7b": 5, "7nm": 2, "8": [1, 3, 6], "8b": 5, "8x8x1": 1, "9": 1, "93": 6, "A": [1, 2, 6, 8], "And": [3, 4, 5, 6, 8], "As": 3, "At": 2, "Be": 6, "But": 7, "By": [1, 6], "For": [0, 2, 6], "If": [1, 2, 3, 5, 6, 7], "In": [1, 2, 5, 6], "It": [2, 6, 7], "Not": 2, "The": [0, 1, 2, 3, 5], "Then": [0, 4], "These": [2, 3, 6], "To": [1, 2, 5, 6, 7], "With": 3, "__float2bfloat16": 6, "__global__": [1, 6], "__grid_constant__": 6, "__launch_bounds__": 6, "__shared__": 6, "__shfl_sync": 2, "__shm": 6, "__syncthread": [2, 6], "__syncwarp": 6, "_col": 3, "_model_config": 5, "_row": 3, "abl": 0, "about": [0, 3, 6], "abov": [0, 6], "acceler": [2, 5], "access": [0, 1, 2, 6], "accum": 0, "accumul": 6, "accuraci": 2, "achiev": [4, 6], "across": [0, 1, 2, 6], "act": 2, "action": 2, "activ": 2, "actual": 6, "ad": [3, 5], "add": [2, 3, 4], "addarrai": 1, "addit": [1, 2], "addnumb": 1, "address": 8, "adjac": 2, "adjust": [2, 6], "advanc": 6, "after": 7, "afterward": 6, "again": 2, "aggress": 7, "ai": 2, "ain": 8, "al": 6, "alg": 6, "algorithm": 0, "alia": 6, "align": [2, 6], "alignment_dummi": 6, "all": [3, 6, 7], "alloc": [0, 1, 6], "allow": [1, 6], "along": 3, "alreadi": 2, "also": [1, 2, 3, 6], "altern": [2, 7], "altogeth": 6, "alwai": [0, 2], "am": [3, 6], "amd": 6, "america": 5, "among": 2, "amongst": [0, 6], "amper": 2, "an": [0, 1, 3, 6], "analog": 6, "ani": [0, 2, 3, 4, 5, 6], "anoth": 2, "anyth": [6, 7], "api": 2, "appropri": 0, "apt": 7, "ar": [0, 1, 2, 3, 5, 6, 7], "architectur": 5, "argument": [1, 6], "around": 8, "arrai": 1, "arriv": 6, "art": 5, "aspect": [3, 6], "assembli": 6, "assign": [2, 6], "associ": 6, "async": 0, "asynchron": [0, 6, 8], "atent": 6, "att_block": 6, "att_block_mma": 6, "attend_k": 6, "attent": [4, 6, 8], "attn": 5, "attn_til": 6, "auto": 6, "automat": 2, "avail": [2, 6], "avoid": [1, 2, 6], "b": [1, 2, 3, 6], "backend": 0, "backward": 5, "balanc": 2, "bank": [0, 2, 8], "base": [1, 2, 6], "bash": 5, "basic": [0, 2], "batch": [3, 6], "batch_id": 3, "becaus": [2, 6, 7, 8], "been": 6, "befor": [0, 1, 2, 6, 7], "begin": 6, "behavior": 6, "being": [2, 6], "benchmark": 1, "benefit": 6, "better": [2, 6], "between": [0, 2], "bf16": [2, 3, 6], "bfloat16": 6, "big": 0, "bin": 7, "bind": 7, "bit": 6, "blackwel": 2, "blend": 2, "block": [0, 6], "blockdim": 1, "blockidx": [1, 3, 6], "blockiidx": 3, "blocksiz": 1, "blog": 6, "blogpost": 6, "blur": 2, "boil": 2, "boost": 2, "both": [2, 6], "bottleneck": 2, "bound": 1, "boundari": 2, "bracket": 1, "branch": 2, "break": 2, "bring": [1, 6], "broader": 6, "brrr": 6, "bug": [6, 7], "build": [5, 6, 7, 8], "built": [2, 8], "bunch": 7, "busi": [0, 2], "byte": 2, "c": [1, 3, 4, 6, 7], "cach": [0, 1], "calcul": 2, "call": [0, 2, 3, 4, 6, 8], "can": [0, 1, 2, 3, 4, 6, 7, 8], "capac": 2, "capit": 5, "card": 1, "care": 6, "carefulli": [2, 6], "carri": 0, "case": [1, 2], "cast": 6, "catastroph": 6, "caus": 2, "causal": 5, "cd": 5, "certain": [3, 6], "challeng": 2, "chanc": 7, "chang": 1, "channel": [6, 8], "check": [4, 6], "checkout": 5, "chef": 2, "chip": 2, "choos": 2, "chop": 2, "chunk": [2, 3, 6], "clang": 7, "clean": 4, "cli": 5, "clock": 2, "clone": 7, "cluster": 2, "coalesc": [2, 6], "code": [0, 1, 2, 3, 4, 6], "coexist": 2, "col": 6, "col_l": 6, "col_vec": 6, "collabor": [2, 6], "collect": 1, "color": 2, "column": 6, "com": 5, "combin": [2, 5], "come": 6, "comment": 6, "common": [0, 6], "commonli": 1, "commun": 6, "compat": [6, 7], "compil": [3, 6, 7], "complementari": 0, "complet": [1, 6], "complex": 2, "complic": 6, "compon": 1, "comprehens": 6, "comput": [0, 1, 2, 6, 8], "concept": [1, 7], "concurr": 2, "conda": 7, "config": [5, 7], "configur": [1, 2], "conflict": [0, 2, 8], "consecut": 2, "consequ": [6, 8], "const": 6, "constant": 1, "constexpr": 6, "consum": 0, "contain": 2, "content": 6, "context": 5, "continu": 1, "contrast": 6, "contribut": 6, "control": 2, "convert": [5, 6], "convolut": [0, 4, 6], "cooper": 1, "coordin": 2, "copi": [1, 6, 8], "core": [0, 8], "corollari": 0, "correct": 0, "correspondingli": 6, "could": 6, "counter": 1, "countless": 1, "cover": [1, 2], "cpp": 4, "cpu": [1, 2], "creat": [1, 3, 4, 6], "creation": 0, "crippl": 2, "critic": 2, "crucial": 0, "crunch": 2, "cu": 4, "cube": 1, "cubla": 2, "cuda": [3, 4, 6, 7, 8], "cuda_hom": 7, "cudafre": 1, "cudamemcpydevicetodevic": 1, "cudamemcpydevicetohost": 1, "cudamemcpyhosttodevic": 1, "cuh": [6, 7], "current": [1, 2, 5], "cut": 2, "cuter": 6, "cycl": [2, 6], "d": [3, 5, 6], "d_a": 1, "d_b": 1, "d_c": 1, "data": [0, 1, 2, 3, 6, 8], "dataset": 1, "date": 7, "decid": 2, "decis": 2, "declar": 6, "decod": 6, "decreas": 0, "deep": 8, "deeper": 4, "defaukt": 1, "default": 6, "defin": [0, 1, 3], "definit": 1, "delai": 2, "deltanet": 5, "demo_8b": 5, "depend": [0, 2, 6], "descriptor": 6, "design": 2, "despit": 6, "destin": 6, "detail": [0, 6], "determin": 1, "dev": 7, "develop": [0, 4], "devic": 1, "device_arrai": 1, "differ": [0, 1, 2, 3, 6], "differenti": 0, "dimens": [1, 3], "direct": 1, "directli": 2, "directori": [5, 7], "discord": [6, 8], "discuss": [0, 3], "dispatch": [3, 4], "distribut": 8, "div_row": 6, "dive": 4, "diverg": 2, "divid": 2, "do": [0, 2, 6, 7, 8], "doc": 7, "document": 6, "doe": [0, 3, 6, 7], "doesn": [1, 6], "don": [0, 1, 2, 3, 6], "dot": 6, "doubl": [2, 6], "down": [0, 2, 6], "download": 5, "downsid": 2, "dp": 2, "dram": [2, 6], "dst": 3, "due": [2, 7], "dummi": 4, "dure": 1, "dynam": 2, "e": [0, 1, 2, 3, 5, 6], "each": [0, 1, 2, 3, 6, 8], "earlier": 7, "easi": [6, 7, 8], "easier": 6, "ecc": 2, "edg": [1, 2, 6], "effect": 6, "effici": [1, 2, 8], "element": [1, 6], "els": [2, 6], "emb": 8, "enabl": 8, "encount": 6, "end": 7, "enough": [0, 1], "ensur": 2, "enter": 5, "entir": 1, "env": [5, 7], "environ": [5, 7], "equal": 6, "error": [6, 7], "especi": 8, "essenti": 1, "etc": 6, "even": [6, 8], "eventu": 8, "everi": [2, 6], "everyon": 6, "exact": 2, "exampl": [1, 2, 3, 6, 7], "except": [3, 6], "excit": 6, "execut": [0, 1], "exist": 6, "exp2": 6, "expect": 6, "experi": [3, 7], "explan": 6, "explicitli": 2, "exponenti": 6, "export": 7, "extens": [7, 8], "extern": 6, "face": 5, "fail": 6, "failur": 6, "fairli": [6, 7, 8], "familiar": 0, "faq": 0, "fast": [1, 2, 8], "faster": [2, 6], "fastest": 1, "featur": [0, 2, 5, 6], "feel": 6, "fermi": 2, "fetch": 2, "few": [6, 8], "fewer": 2, "fft": [0, 4], "file": [4, 5, 7], "filter": 2, "final": 7, "find": [6, 7], "finfet": 2, "finish": [0, 1, 8], "first": [2, 6], "fit": [1, 2], "fix": 2, "flash": [5, 6, 8], "flashattent": 6, "flavor": 6, "flight": 6, "float": [2, 3, 6], "flow": 0, "flush": 0, "focu": [0, 2], "folder": [3, 4, 6, 7], "follow": [1, 2, 3, 6], "format": 2, "formula": 1, "forward": 5, "four": 3, "fp": 2, "fp16": 2, "fp32": [2, 3], "fp4": 2, "fp64": 2, "fp8": [2, 3, 6], "framework": 8, "free": [0, 1], "frequent": [1, 2], "from": [0, 1, 2, 3, 4, 5, 6, 7, 8], "function": [0, 2, 3, 4, 6, 8], "futur": 1, "g": [0, 1, 2, 3, 5, 6, 7], "gb": 2, "gcc": 7, "gddr5": 2, "gddr5x": 2, "gddr6": 2, "gddr6x": 2, "gddr7": 2, "gen": 2, "gen4": 2, "gener": [0, 6, 8], "gentest": 4, "get": [0, 2, 4, 7, 8], "gg": [6, 8], "git": 5, "github": 5, "give": [1, 6], "given": [3, 6], "gl": 6, "global": [0, 3, 6], "global_layout": 6, "globalthreadid": 1, "go": [1, 6, 7], "goe": 6, "good": 6, "got": 8, "gpu": [1, 4, 6, 8], "gpumod": [6, 8], "gqa": 5, "graphic": [1, 2], "great": [0, 2], "grid": [2, 3, 6], "griddim": 1, "gridsiz": 1, "group": [1, 2, 6], "group_warp": 6, "groupid": 6, "h": [3, 6], "h100": [0, 6, 7, 8], "ha": [1, 2, 3, 6, 7], "half": [2, 3, 6], "handl": [1, 2], "happen": 6, "har": 4, "hard": 6, "hardwar": [2, 6, 8], "hassl": 2, "have": [1, 2, 3, 6], "hazyresearch": 4, "hbm": [0, 3], "hbm2": 2, "hbm3": 2, "hbm3e": 2, "head": [3, 6], "head_id": 3, "header": 7, "hedgehog": 5, "height": [3, 6], "help": [0, 2, 4, 6], "here": [0, 1, 2, 3, 4, 5, 6, 8], "hide": 8, "high": [2, 8], "higher": 2, "highli": 2, "hold": [0, 2, 6], "hood": [3, 4, 6, 8], "hope": 0, "hopper": 2, "host": 1, "host_arrai": 1, "how": [0, 1, 2, 3, 4, 6], "howev": [0, 6], "http": [5, 6, 8], "hug": 5, "huggingfac": 5, "hurt": 2, "hyper": 2, "i": [0, 1, 2, 3, 5, 6, 7, 8], "id": 6, "idea": 0, "ideal": 6, "identifi": 1, "ignor": 6, "imag": [1, 2], "immedi": 2, "impl": 4, "implement": 8, "implicit": 6, "import": [0, 6], "improv": 6, "includ": [0, 3, 5, 6, 7, 8], "incom": 3, "independ": [1, 2], "index": 3, "individu": [0, 2, 6], "infer": [2, 5, 6], "influenc": 0, "ing": 6, "initi": 6, "input": [0, 3], "instal": [5, 6], "instanc": 3, "instant": 2, "instanti": [0, 3, 6], "instantli": 2, "instead": [0, 1], "instruct": [1, 2, 6], "int": [1, 2, 6], "int32": 2, "int4": 2, "int8": 2, "integ": 2, "integr": 5, "intens": 5, "interact": 6, "interconnect": 2, "intermedi": 1, "intern": 6, "interpol": 2, "invit": [6, 8], "invoc": 1, "involv": 8, "issu": [6, 7], "iter": 6, "its": [0, 1, 2, 6], "itself": [7, 8], "j": 6, "jag": 2, "join": [6, 8], "jordan": 7, "juravski": 7, "just": [0, 3, 4, 6, 7], "k": 6, "k_reg": 6, "k_smem": 6, "kb": 2, "keep": 2, "kei": 8, "kepler": 2, "kernel": [2, 3, 5, 6], "kg": 6, "kitten": [3, 6, 7], "know": [1, 6], "known": 6, "kv_block": 6, "kv_idx": 6, "l": 6, "l2": [0, 8], "languag": 6, "larg": [1, 5], "larger": 0, "largest": 1, "last": 8, "latenc": 8, "later": 6, "launch": [1, 3, 6], "layer": [2, 4], "layernorm": 6, "layout": 6, "ld_library_path": 7, "learn": [3, 5, 8], "least": 8, "leav": 6, "length": 3, "less": 6, "let": [2, 3, 4, 6], "level": [2, 6], "lg2": 6, "lib64": 7, "librari": [2, 6], "like": [0, 1, 2, 3, 4, 6, 7], "limit": [1, 2], "line": [0, 1, 6], "linear": 5, "link": [6, 8], "live": 6, "ll": [0, 3, 6, 7], "llama": [5, 6], "llama_demo": 5, "llm": [5, 6], "load": [0, 2, 3, 4, 6, 8], "load_async": 6, "load_async_wait": 6, "load_block": 6, "load_group": 6, "loadid": 6, "local": [1, 6, 7], "locat": [1, 2], "lockstep": 2, "logic": 0, "login": 5, "lolcat": 6, "lolcats_demo": 5, "long": 6, "look": [3, 6], "lookup": 1, "loop": [1, 2], "lot": [2, 6], "love": 5, "low": 2, "lower": 2, "m": 6, "machin": [3, 8], "mai": 6, "main": [1, 2, 4], "make": [1, 2, 3, 4, 5, 6, 8], "makefil": 4, "malloc": 1, "mamba": 0, "manag": [0, 2], "mani": [0, 1, 2, 6, 7], "manipul": [6, 8], "manual": 2, "manycor": 8, "map": [0, 6], "massiv": 2, "match": [2, 4], "math": 2, "matmul": [0, 3, 6], "matric": 2, "matrix": [0, 2, 6, 8], "matter": 2, "max": [2, 3, 6], "max_vec": 6, "max_vec_last": 6, "maxim": [1, 2], "maxwel": 2, "mean": [2, 3], "medic": 2, "memori": [0, 3, 6, 8], "method": 5, "mfma": 6, "micro": 4, "might": [1, 3, 6], "mind": 6, "minim": 2, "minimis": 1, "minut": 6, "mix": 2, "mma_ab": 6, "mma_abt": 6, "mode": [0, 2, 6, 8], "model": 5, "modern": [7, 8], "moment": 2, "monei": 7, "more": [0, 2, 3, 8], "most": [2, 3, 6, 7], "move": 2, "movement": 6, "mp": 8, "much": [2, 6, 7], "mul": 6, "mul_row": 6, "multipl": [0, 2, 6], "multipli": [0, 2, 6, 8], "multiprocessor": 2, "must": [2, 6], "my": 3, "myself": 0, "n": [2, 3, 6], "naiv": 6, "name": [1, 6], "namespac": 6, "nativ": 8, "nearbi": 2, "need": [0, 1, 2, 3, 4, 6, 7, 8], "neg_infti": 6, "neighbor": 2, "network": 2, "neural": 2, "new": 6, "newer": 2, "next": [2, 5, 6], "next_load_idx": 6, "next_tic": 6, "nice": 0, "nicest": 7, "nine": 8, "nineti": 8, "node": 2, "non": [2, 5], "norm": [4, 6], "norm_vec": 6, "normal": 6, "nov": 6, "now": [2, 3, 4, 6], "nuke": 6, "num_work": 6, "number": [0, 1, 2, 3, 5, 6, 7], "nvcc": 7, "nvidia": 2, "nvlink": 2, "o": [6, 8], "o_reg": 6, "object": [3, 6], "occup": [0, 2], "oct": 6, "off": 2, "offer": 8, "often": [6, 8], "og": 6, "old": 6, "older": 2, "onboard": 6, "onc": 2, "one": [1, 2, 8], "onli": [1, 2, 6, 7], "onto": 6, "op": 6, "oper": [2, 3, 6], "operand": 6, "optim": 2, "option": 0, "order": 0, "organ": [1, 2], "orient": [6, 7], "origin": 6, "orthogon": 6, "other": [1, 2, 8], "otherewis": 6, "our": [1, 3, 4, 6, 7, 8], "out": [0, 2, 3, 5, 6, 7], "outer": 6, "output": [0, 6], "outsid": 2, "over": [2, 3, 6], "overal": [0, 1, 2], "overlap": [2, 8], "overload": 2, "overview": 2, "own": [1, 2, 3, 6], "pad": 2, "pain": 6, "pair": [2, 6], "paper": 6, "parallel": [1, 2, 3, 6], "paramet": 1, "parameter": 6, "part": 3, "partial": 2, "particular": 6, "particularli": 6, "pascal": 2, "pass": [1, 3, 6], "past": 1, "path": [2, 7], "pattern": [0, 2, 6], "pcie": 2, "peak": [0, 4], "per": [1, 2, 6], "perf": 2, "perfect": 2, "perform": [0, 1, 2, 4], "perhap": 0, "persist": 0, "person": 2, "physic": 2, "pick": 2, "piec": 2, "pip": 5, "pipelin": [6, 7], "pixel": [1, 2], "place": [0, 6], "plai": 7, "pleas": [5, 6], "point": [2, 3, 6, 7, 8], "pool": [0, 2], "poorli": 2, "posit": [1, 3], "possibl": [1, 6, 7], "potenti": [3, 6], "power": 5, "pr": [5, 6], "pragma": 6, "pre": 6, "precis": [2, 6], "prefil": 5, "present": 3, "pretti": 6, "primit": 6, "principl": 8, "printf": 1, "prior": 0, "priorit": 2, "privat": [1, 2], "probabl": [0, 7], "problem": [2, 8], "process": [1, 2], "processor": [2, 8], "produc": 0, "product": 6, "proger": 5, "program": [1, 2, 3], "programm": 2, "progress": 6, "promis": 6, "prompt": 5, "properli": [1, 2], "protect": 6, "provid": [1, 3, 6, 7], "pull": 0, "pure": 6, "purpos": 1, "put": 7, "py": [4, 5, 7], "python": [3, 4, 5, 7], "pytorch": [3, 4, 6, 7], "q": [2, 6], "q_reg": 6, "q_seq": 6, "qg": 6, "qkvo_til": 6, "qo_smem": 6, "quadrat": 5, "quick": 7, "quickli": [1, 2], "qwen": [5, 6], "randn": 4, "rang": 1, "raw": 3, "re": [0, 2, 3, 6], "reach": 5, "read": [1, 2, 6], "readi": [2, 6], "readm": 6, "realli": [0, 1, 6, 8], "realloc": 0, "recal": 5, "recent": 5, "recommend": 6, "reduc": [2, 6], "reduct": [2, 6], "region": 1, "regist": [0, 1, 3, 6], "register_til": 6, "regular": 1, "reinterpret_cast": 6, "relev": 3, "render": 2, "repeatedli": 2, "repo": 7, "report": 6, "repres": [1, 3], "request": 2, "requir": [0, 2, 6], "rescal": 6, "resourc": 2, "respons": 2, "restructur": 0, "result": [1, 2, 4, 6], "reus": [0, 2, 6], "review": 6, "right": [6, 8], "risc": 6, "rocm": 8, "root": 7, "rotari": 0, "row": 6, "row_l": 6, "row_max": 6, "row_sum": 6, "rt": [2, 6], "rt_bf": 6, "rt_fl": 6, "rtx": [2, 6], "rule": 0, "run": [0, 1, 2, 3, 4, 5, 6, 7, 8], "runtim": [3, 6], "same": [0, 1, 2, 6], "save": 4, "scan": [5, 6], "scientif": 2, "scratch": [0, 8], "scratchpad": 2, "second": 6, "section": [0, 1, 6], "see": [0, 3], "select": [5, 7], "separ": [2, 6], "sequenc": [3, 4], "sequenti": 6, "seri": 5, "serial": 7, "set": [0, 1, 2, 3, 4, 5, 6, 7], "setup": [0, 3, 7], "sever": [1, 5], "sh": 5, "shape": 3, "share": [0, 1, 3, 6, 8], "shared_alloc": 6, "shared_til": 6, "sharp": 6, "sheet": 1, "short": 5, "should": [0, 2, 3, 6, 8], "show": [1, 3, 4, 6], "shown": [0, 3], "shut": 0, "signatur": 6, "silent": 6, "silicon": 8, "similar": 1, "similarli": 6, "simpl": [1, 4, 6, 8], "simpler": 2, "simpli": 6, "simplic": [6, 8], "simplifi": 2, "simul": 2, "simultan": 2, "sinc": 0, "singl": [1, 2, 6], "sit": 6, "size": [0, 1, 2, 3, 5, 6], "size_t": 6, "sizeof": 1, "skip": 2, "slave": 7, "slide": 5, "slightli": 6, "slot": 2, "slow": [1, 2], "slower": 2, "slowest": 1, "sm": 6, "small": [1, 2, 6, 8], "smaller": 8, "smallest": 1, "smooth": 2, "so": [1, 2, 3, 6, 7, 8], "softmax": 6, "solv": 2, "some": [2, 3, 4, 6], "sometim": [0, 1], "soon": 8, "sourc": [5, 6, 7], "space": [1, 2], "sparsiti": 2, "spatial": 2, "speak": 8, "spec": 2, "special": [1, 6], "specif": [0, 1, 2], "specifi": [1, 3, 4, 6], "speed": [1, 2, 8], "spill": [1, 2], "split": [2, 6], "src": [3, 5, 6, 7], "st_bf": 6, "st_fl": 3, "st_hf": 6, "stage": 6, "stai": 6, "stall": 2, "stand": 6, "start": [0, 1], "starter": 6, "state": [0, 5], "statement": 2, "static": 6, "steal": 0, "step": 2, "still": 6, "store": [0, 1, 2, 3, 6, 8], "stream": [1, 2], "struct": [3, 6], "structur": 2, "stuck": 2, "stuff": 7, "stupidli": 8, "style": 3, "sub": 6, "sub_row": 6, "subtil": 6, "subtleti": 6, "subtract": 6, "sudo": 7, "suit": [3, 6], "sum": [2, 3, 6], "support": [2, 3, 6], "sure": [1, 4, 6], "sustcsonglin": 5, "switch": 2, "swizzl": 0, "sync": 2, "synchron": [1, 2], "syntax": 3, "system": 2, "t": [0, 1, 2, 3, 6, 7, 8], "tabl": 1, "tail": 6, "take": [2, 3, 6], "talk": 0, "task": [0, 1, 2, 5, 6], "tb": 2, "team": [1, 2], "teamwork": 2, "tech": 2, "tell": 8, "temperatur": 6, "templat": [0, 3, 4, 6, 8], "temporari": [1, 2], "tensor": [0, 3, 4, 8], "term": 6, "tflop": 6, "than": [2, 6, 8], "thank": 7, "theater": 6, "thei": [0, 2, 6, 8], "them": [2, 4, 6], "themselv": 7, "theoret": 6, "thi": [0, 1, 2, 3, 4, 6, 8], "thing": [2, 4, 8], "think": [0, 1, 2, 3, 6, 8], "those": [7, 8], "thousand": [2, 6], "thread": [0, 6], "threadidx": 1, "three": [0, 4, 6, 8], "through": [0, 1, 4, 6], "throughput": [0, 2], "thunderkitten": [4, 5, 7], "thunderkittens_root": 7, "thundermitten": 6, "tic": 6, "tile": [0, 2, 3, 6], "time": [0, 1, 2, 3, 6], "tip": 2, "tk": [6, 7], "tma": [6, 8], "togeth": [2, 7], "told": 6, "too": [2, 6, 8], "tool": 2, "torch": 3, "touch": 6, "trade": 2, "train": [2, 5, 6], "transact": 2, "transform": 5, "transpos": 6, "transpose_sep": 6, "tri": 6, "tricki": 8, "try": [3, 5], "tsmc": 2, "tupl": 3, "ture": 2, "tweak": 2, "two": [0, 1, 2, 3, 6], "txt": 4, "type": [0, 2, 3], "typenam": 6, "typic": 0, "u": [4, 5, 6, 8], "ultra": 2, "unclear": 6, "undefin": 6, "under": [3, 4, 6, 8], "underli": 6, "understand": [0, 2, 3, 4, 6], "unfortun": 6, "unifi": 2, "uniqu": 1, "unit": [1, 6], "unlik": 2, "unnecessarili": 2, "unrol": 6, "until": 1, "up": [2, 3, 4, 6, 7, 8], "updat": 7, "uplevel": 3, "us": [0, 1, 2, 3, 5, 6, 7, 8], "usag": 2, "usr": 7, "util": 8, "v": [0, 6], "v_reg": 6, "v_smem": 6, "valid": 6, "valu": [0, 2, 4, 8], "variabl": [1, 5, 7], "variant": 5, "ve": [6, 7], "vec": 6, "vector": [1, 6], "veget": 2, "veri": [0, 1, 2, 6], "version": 7, "vg": 6, "via": 2, "void": [1, 6], "volta": 2, "w": [2, 3], "wai": [0, 1, 2, 8], "wait": [1, 2], "walk": [0, 4], "want": [0, 3, 5, 6, 7, 8], "warn": 6, "warp": [0, 6], "warp_thread": 6, "warpgroup": [0, 6], "warpid": 6, "wast": 2, "water": 2, "we": [0, 1, 3, 4, 5, 6, 7, 8], "weird": 7, "well": 2, "wgmma": [6, 7, 8], "what": [0, 2, 3, 6, 8], "when": [0, 1, 2, 3, 6], "where": [0, 6, 7, 8], "wherea": 6, "wherev": 6, "whether": 4, "which": [0, 1, 2, 3, 5, 6], "while": [1, 2, 6], "who": [0, 1], "why": 2, "wide": 2, "width": 3, "window": 5, "wise": [1, 6], "wish": 6, "within": [1, 2], "without": [2, 6], "wmma": 2, "won": 8, "word": 6, "work": [0, 1, 2, 4, 6, 7, 8], "worker": [0, 1, 2, 6, 8], "workerid": 6, "workload": 2, "workspac": 2, "worri": [0, 3], "would": 6, "wrangl": 0, "wrapper": [3, 6], "write": [0, 2, 6, 8], "written": [3, 6, 8], "wrong": 7, "x": [1, 3, 6], "y": [1, 3, 6], "yaml": 5, "ye": 0, "year": 8, "you": [0, 1, 3, 5, 6, 7, 8], "your": [0, 1, 3, 6, 7, 8], "yourself": [6, 8], "z": [1, 6], "zero": 6, "zoom": 2}, "titles": ["Advanced Overview", "Nuts and Bolts of the operations", "SM", "TK Basic Overview", "Trying out your kernel", "TK Demos: play with kittens!", "Getting started", "Installation", "ThunderKittens"], "titleterms": {"": 6, "1": 2, "10": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 2, "9": 2, "It": 1, "advanc": 0, "all": [1, 2], "an": 2, "architectur": 2, "attent": 5, "base": 5, "basic": [3, 4], "best": 2, "block": [1, 2], "bolt": 1, "cach": 2, "calcul": 1, "comparison": 2, "compon": 2, "core": 2, "cuda": [1, 2], "cudadevicesynchron": 1, "cudamalloc": 1, "cudamemcpi": 1, "demo": [5, 6], "dim3": 1, "exampl": [0, 4], "execut": 2, "exist": 7, "fastest": 2, "function": 1, "gener": 5, "get": 6, "global": [1, 2], "gpu": 2, "grid": 1, "helper": 1, "hide": 2, "hierarchi": 2, "id": 1, "index": 1, "instal": 7, "involv": 6, "kei": 2, "kernel": [0, 1, 4, 7, 8], "kitten": 5, "l1": 2, "latenc": 2, "learn": 6, "librari": 7, "lolcat": 5, "manag": 1, "manual": 6, "memori": [1, 2], "model": 6, "more": 6, "nut": 1, "nvidia": 6, "oper": 1, "other": 6, "out": 4, "overview": [0, 3], "plai": 5, "practic": 2, "pre": 7, "primit": 8, "program": 6, "put": 1, "regist": 2, "requir": 7, "restrict": 6, "schedul": 2, "scope": 6, "setup": 5, "share": 2, "slowest": 2, "sm": 2, "some": 1, "special": 2, "speedi": 8, "start": 6, "tensor": 2, "test": 6, "textur": 2, "theori": 1, "thread": [1, 2], "thunderkitten": [6, 8], "tile": 8, "tk": [0, 3, 4, 5], "togeth": 1, "try": 4, "type": [1, 6], "understand": 1, "unit": 2, "warp": 2, "your": [4, 5]}})