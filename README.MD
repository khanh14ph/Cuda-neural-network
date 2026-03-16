# CUDA Neural Network

A fully-connected neural network trained on MNIST digit classification, implemented across four computational backends for performance comparison.

## Architecture

3-layer MLP: **784 → 128 → 256 → 10**

- Input: 28×28 flattened MNIST images
- Hidden layers: ReLU activation
- Output: Softmax (10 digit classes)
- Loss: Cross-entropy
- Optimizer: SGD (lr=0.01)
- Batch size: 500, Epochs: 50

## Implementations

| Binary | Backend | Notes |
|---|---|---|
| `main_cblas` | CPU + OpenBLAS | BLAS-accelerated GEMM |
| `main_native` | CPU + OpenMP | Hand-coded tiled GEMM (32×32 tiles) |
| `main_gpu` | CUDA hand-coded | Custom GEMM kernel (16×16 shared-memory tiles) |
| `main_gpu_cublas` | CUDA + cuBLAS | Production-quality GPU path |

## Performance (5 epochs, Volta GPU)

| Implementation | Train Time | Infer Time | Grind Rate | Test Acc |
|---|---|---|---|---|
| CPU BLAS | 2.80s | 0.053s | 89k samples/s | 87.5% |
| CPU Native GEMM | 3.42s | 0.064s | 73k samples/s | 88.4% |
| GPU Hand-coded | 4.18s | 0.023s | 717k samples/s | — |
| GPU cuBLAS | 4.19s | 0.017s | 597k samples/s | 94.5% |

GPU inference is ~6× faster than CPU BLAS. Full 50-epoch cuBLAS training reaches **94.45% test accuracy**.

## File Structure

```
├── main.cpp          # CPU training/inference pipeline
├── main.cu           # GPU training/inference pipeline
├── layer.h           # CPU layer struct (forward pass, He init)
├── layer_gpu.cuh     # GPU layer struct (device memory management)
├── kernels.cuh       # CUDA kernels (GEMM, ReLU, softmax, SGD)
├── data.h            # MNIST binary file loader
├── Makefile
└── benchmark.sh      # SLURM job script
```

## Prerequisites

- g++ with C++11
- OpenBLAS
- NVIDIA CUDA Toolkit (`nvcc`)
- MNIST dataset files:
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`
  - `t10k-images-idx3-ubyte`
  - `t10k-labels-idx1-ubyte`

Update the data path in `main.cpp` / `main.cu` if needed.

## Build

```bash
make          # builds all 4 binaries
make clean    # remove binaries
```

The GPU targets compile for `sm_70` (Volta). Edit the `Makefile` to target a different architecture (e.g., `sm_75`, `sm_80`, `sm_90`).

## Run

```bash
./main_cblas          # CPU + OpenBLAS
./main_native         # CPU + hand-coded GEMM
./main_gpu            # GPU + hand-coded kernels
./main_gpu_cublas     # GPU + cuBLAS
```

### Cluster (SLURM)

```bash
sbatch benchmark.sh   # runs all variants and saves output to slurm/
```
