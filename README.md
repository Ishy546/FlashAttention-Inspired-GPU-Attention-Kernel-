# flashattn_lite

FlashAttention-inspired attention forward kernels in CUDA/C++ with a PyTorch extension.

## What's here

- `attention_ref.py` – PyTorch reference (materializes the full N×N score matrix).
- `attention_naive` – baseline: QKᵀ → softmax → PV as 3 separate kernels, with S/P in global memory.
- `attention_flash` – tiled kernel: K/V tiles in shared memory, online softmax, no N×N matrix, causal tile skipping.

## Layout

```
.
├── attention_ref.py        # PyTorch reference implementation
├── setup.py                # builds the flashattn_lite extension
├── test_bench.py           # correctness checks + benchmark
└── csrc/
    ├── attention.cpp       # C++ binding layer (input validation + pybind11)
    └── attention_kernels.cu# CUDA kernels (naive + tiled/flash)
```

## Requirements

- An actual **NVIDIA GPU** (the kernels run on CUDA only — there's no CPU fallback).
- The **CUDA toolkit** (`nvcc`) matching your PyTorch build.
- A **CUDA-enabled PyTorch** install.

This cannot be built or run on a machine without an NVIDIA GPU. If you don't have
one locally, use Google Colab with a GPU runtime (see below).

## Build & run (local, with a GPU)

```bash
pip install -e . --no-build-isolation   # or: python setup.py build_ext --inplace
python test_bench.py --N 2048 --D 64
```

## Build & run on Google Colab

1. Open [https://colab.research.google.com](https://colab.research.google.com) and create a new notebook.
2. Switch to a GPU runtime: **Runtime → Change runtime type → Hardware accelerator: GPU → Save**.
3. Confirm a GPU is attached:

   ```python
   !nvidia-smi
   ```

4. Get the code into the Colab session. Either clone it:

   ```python
   !git clone <your-repo-url> flashattn_lite
   %cd flashattn_lite
   ```

   or upload the folder (with `csrc/`, `setup.py`, `attention_ref.py`, `test_bench.py`)
   via the file browser, then `%cd` into it.

5. Build the extension (Colab already ships CUDA-enabled PyTorch and `nvcc`):

   ```python
   !pip install -e . --no-build-isolation
   ```

6. Run the benchmark. Start with modest sizes to keep GPU usage light:

   ```python
   !python test_bench.py --N 1024 --D 64
   ```

   Bump `--N` (e.g. 2048) once you've confirmed it builds and passes. The script
   runs correctness checks for both the naive and flash kernels (causal and
   non-causal, including an odd N that exercises tile/row bounds) and then prints a
   small timing/peak-memory table against the PyTorch reference and `F.scaled_dot_product_attention`.

Constraints: float32, contiguous `[B, H, N, D]`, `D ∈ {32, 64, 128}`.

## Ideas to extend

- fp16/bf16 + tensor cores (WMMA/MMA) for QKᵀ and PV.
- Tune BR/BC; split D across threads to cut register pressure at D=128.
- Save logsumexp and write the backward pass.
- Profile with Nsight Compute (shared-mem throughput, occupancy, HBM traffic).


Side note: You need an actual Nvidia GPU to run this. You can run it for free on google colab. 