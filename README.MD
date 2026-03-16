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
| `main_gpu_cublas` | CUDA + cuBLAS | cuBLAS via `USE_CUBLAS` flag; `my_gemm()` wraps `cublasSgemm` with row-major↔column-major swap |

## Performance (50 epochs)

| Version | Processor | Accuracy | Grind Rate (samples/s) | Training Time (s) | TPB / CPU cores |
|---|---|---|---|---|---|
| GPU Native | V100 | 94.18% | 584,046 | 4.2 | 256 |
| GPU cuBLAS | V100 | 94.28% | 543,609 | 4.5 | 256 |
| CPU Native | Caslake | 94.17% | 92,763 | 26.95 | 8 |
| CPU BLAS | Caslake | 94.11% | 139,653 | 17.90 | 8 |

GPU native achieves the highest grind rate (~584k samples/s), ~6.3× faster than CPU BLAS. All four backends reach ~94% test accuracy after 50 epochs.

**Anomalous behavior:** GPU cuBLAS is slightly slower than GPU native (543k vs 584k samples/s, 4.5s vs 4.2s training time). This is likely due to overhead from the cuBLAS handle and the row-major→column-major transpose at each GEMM call outweighing cuBLAS's kernel efficiency at this matrix size.

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
