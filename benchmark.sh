#!/bin/bash
#SBATCH --job-name=job-info
#SBATCH --account=mpcs51087
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=rtx6000
#SBATCH --output=/home/khanhnd/HPC/project-3-winter-2026-khanh14ph/final/slurm/%j.%N.stdout
#SBATCH --error=/home/khanhnd/HPC/project-3-winter-2026-khanh14ph/final/slurm/%j.%N.stderr
#SBATCH --chdir=/home/khanhnd/HPC/project-3-winter-2026-khanh14ph/final
module load openblas
module load cuda


echo "=== BLAS (cblas_sgemm) ==="
g++ -std=c++11 -Wall -O3 -march=native -ffast-math -fopenmp -DUSE_BLAS main.cpp -o main_cblas -lopenblas
OMP_NUM_THREADS=8 ./main_cblas

echo "=== Hand-coded tiled GEMM ==="
g++ -std=c++11 -Wall -O3 -march=native -ffast-math -fopenmp main.cpp -o main_native  -lopenblas
OMP_NUM_THREADS=8 ./main_native


echo "=== GPU hand-coded GEMM ==="
nvcc -O3 -arch=sm_75 main.cu -o main_gpu
./main_gpu

echo "=== GPU cuBLAS GEMM ==="


nvcc -O3 -arch=sm_75 -DUSE_CUBLAS main.cu -o main_gpu_cublas -lcublas
./main_gpu_cublas