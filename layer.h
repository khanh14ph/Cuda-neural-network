#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <omp.h>
#ifdef USE_BLAS
#include <cblas.h>
#endif
#define TILE_SIZE 32

using namespace std;

enum class Trans { No, Yes };
// C = beta*C + alpha * op(A) * op(B)
void gemm(Trans transA, Trans transB,
          int M, int N, int K,
          float alpha,
          const float* A,
          const float* B,
          float beta,
          float* C) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor,
                transA==Trans::Yes ? CblasTrans : CblasNoTrans,
                transB==Trans::Yes ? CblasTrans : CblasNoTrans,
                M, N, K,
                alpha,
                A, transA==Trans::No ? K : M,
                B, transB==Trans::No ? N : K,
                beta,
                C, N);
#else
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            float tile_A[TILE_SIZE][TILE_SIZE];
            float tile_B[TILE_SIZE][TILE_SIZE];
            float tile_C[TILE_SIZE][TILE_SIZE];
            int mi = min(TILE_SIZE, M - i);
            int mj = min(TILE_SIZE, N - j);

            for (int ii = 0; ii < mi; ii++)
                for (int jj = 0; jj < mj; jj++)
                    tile_C[ii][jj] = beta * C[(i + ii) * N + (j + jj)];

            for (int k = 0; k < K; k += TILE_SIZE) {
                int mk = min(TILE_SIZE, K - k);

                if (transA == Trans::No) {
                    for (int ii = 0; ii < mi; ii++)
                        for (int kk = 0; kk < mk; kk++)
                            tile_A[ii][kk] = A[(i + ii) * K + (k + kk)];
                } else {
                    for (int ii = 0; ii < mi; ii++)
                        for (int kk = 0; kk < mk; kk++)
                            tile_A[ii][kk] = A[(k + kk) * M + (i + ii)];
                }

                if (transB == Trans::No) {
                    for (int kk = 0; kk < mk; kk++)
                        for (int jj = 0; jj < mj; jj++)
                            tile_B[kk][jj] = B[(k + kk) * N + (j + jj)];
                } else {
                    for (int kk = 0; kk < mk; kk++)
                        for (int jj = 0; jj < mj; jj++)
                            tile_B[kk][jj] = B[(j + jj) * K + (k + kk)];
                }

                for (int ii = 0; ii < mi; ii++)
                    for (int jj = 0; jj < mj; jj++)
                        for (int kk = 0; kk < mk; kk++)
                            tile_C[ii][jj] += alpha * tile_A[ii][kk] * tile_B[kk][jj];
            }

            for (int ii = 0; ii < mi; ii++)
                for (int jj = 0; jj < mj; jj++)
                    C[(i + ii) * N + (j + jj)] = tile_C[ii][jj];
        }
    }
#endif
}

struct Layer {
    int in_size;
    int out_size;
    vector<float> weights;
    vector<float> biases;

    // Cache for backprop
    vector<float> input_cache;
    vector<float> z_cache;

    Layer(int in, int out) : in_size(in), out_size(out) {
        weights.resize(in * out);
        biases.resize(out, 0.0f);

        random_device rd;
        mt19937 gen(rd());
        float std_dev = sqrt(2.0f / in);
        normal_distribution<float> d(0.0f, std_dev);
        for (auto& w : weights) w = d(gen);
    }

    vector<float> forward(const vector<float>& input) {
        int batch_size = input.size() / in_size;
        vector<float> output(batch_size * out_size);
        // broadcast biases into output
        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < out_size; j++)
                output[i * out_size + j] = biases[j];
        gemm(Trans::No, Trans::No,
             batch_size, out_size, in_size,
             1.0f, input.data(), weights.data(),
             1.0f, output.data());
        return output;
    }
};



