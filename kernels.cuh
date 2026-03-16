#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

enum class Trans { No, Yes };

__global__ void he_init_kernel(float* weights, int n, float std_dev, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    weights[idx] = curand_normal(&state) * std_dev;
}

#define TILE 16

// ── GEMM ─────────────────────────────────────────────────────────────────────
// C = alpha * op(A) * op(B) + beta * C  (row-major)
// transA==No : A is M×K → A[m*K + k]
// transA==Yes: A is K×M → A[k*M + m]   (same for B / transB)

__global__ void gemm_kernel(Trans transA, Trans transB,
                             int M, int N, int K,
                             float alpha, const float* A, const float* B,
                             float beta,  float* C)
{
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int k_a = t * TILE + threadIdx.x;
        int k_b = t * TILE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < M && k_a < K)
            ? ((transA == Trans::No) ? A[row * K + k_a] : A[k_a * M + row])
            : 0.0f;

        tile_B[threadIdx.y][threadIdx.x] = (k_b < K && col < N)
            ? ((transB == Trans::No) ? B[k_b * N + col] : B[col * K + k_b])
            : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = beta * C[row * N + col] + alpha * acc;
}

void cuda_gemm(Trans transA, Trans transB,
               int M, int N, int K,
               float alpha, const float* d_A, const float* d_B,
               float beta,  float* d_C)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block>>>(transA, transB, M, N, K,
                                  alpha, d_A, d_B, beta, d_C);
}

// ── Kernels

__global__ void broadcast_bias_kernel(float* out, const float* biases,
                                      int batch_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_size)
        out[idx] = biases[idx % out_size];
}

__global__ void relu_kernel(float* dst, const float* src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        dst[idx] = fmaxf(0.0f, src[idx]);
}

__global__ void relu_grad_kernel(float* delta, const float* z, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        delta[idx] *= (z[idx] > 0.0f ? 1.0f : 0.0f);
}

__global__ void softmax_kernel(float* v, int batch_size, int num_class)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    float* row = v + i * num_class;
    float max_val = row[0];
    for (int j = 1; j < num_class; j++) max_val = fmaxf(max_val, row[j]);
    float sum = 0.0f;
    for (int j = 0; j < num_class; j++) { row[j] = expf(row[j] - max_val); sum += row[j]; }
    for (int j = 0; j < num_class; j++) row[j] /= sum;
}

__global__ void softmax_grad_kernel(float* out, const int* labels,
                                    int batch_size, int num_class)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
        out[i * num_class + labels[i]] -= 1.0f;
}

__global__ void reduce_bias_kernel(const float* delta, float* b,
                                    float scale, int B, int out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= out) return;
    float sum = 0.0f;
    for (int s = 0; s < B; s++) sum += delta[s * out + r];
    b[r] -= scale * sum;
}


__global__ void cross_entropy_kernel(const float* out, const int* labels,
                                     int batch_size, int num_class,
                                     float* d_loss)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    float val = -logf(out[i * num_class + labels[i]] + 1e-7f);
    atomicAdd(d_loss, val);
}

__global__ void argmax_correct_kernel(const float* out, const int* labels,
                                      int batch_size, int num_class,
                                      int* d_correct)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    const float* row = out + i * num_class;
    int best = 0;
    for (int j = 1; j < num_class; j++)
        if (row[j] > row[best]) best = j;
    if (best == labels[i]) atomicAdd(d_correct, 1);
}

