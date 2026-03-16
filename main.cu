#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

#include "layer_gpu.cuh"
#include "kernels.cuh"
#include "data.h"

using namespace std;

#define NUM_CLASS 10
#define blockSize 256

#define CUDA_CHECK(err)                                                     \
    if ((err) != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                cudaGetErrorString(err), __FILE__, __LINE__);               \
        exit(1);                                                            \
    }

#ifdef USE_CUBLAS
static inline void my_gemm(cublasHandle_t handle,
                            Trans transA, Trans transB,
                            int M, int N, int K,
                            float alpha, const float* A, const float* B,
                            float beta,        float* C)
{
    int lda = (transA == Trans::No) ? K : M;
    int ldb = (transB == Trans::No) ? N : K;
    cublasOperation_t opA = (transA == Trans::Yes) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = (transB == Trans::Yes) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}

#else
static inline void my_gemm(int /*unused*/,
                            Trans transA, Trans transB,
                            int M, int N, int K,
                            float alpha, const float* A, const float* B,
                            float beta,        float* C)
{
    cuda_gemm(transA, transB, M, N, K, alpha, A, B, beta, C);
}

#endif

// ── Main ──────────────────────────────────────────────────────────────────────

int main()
{
#ifdef USE_CUBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    // Enable TF32 tensor-core math (Ampere+). Falls back to FP32 on older GPUs.
    // cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cout << "Using cuBLAS for GEMM" << endl;
#else
    int cublas_handle = 0; // unused placeholder
    cout << "Using hand-coded GEMM" << endl;
#endif

    string base = "/home/khanhnd/HPC/project-3-winter-2026-khanh14ph/milestone1/data/";
    MNISTData train_data = load_mnist(base + "train-images-idx3-ubyte",
                                      base + "train-labels-idx1-ubyte");
    MNISTData test_data  = load_mnist(base + "t10k-images-idx3-ubyte",
                                      base + "t10k-labels-idx1-ubyte");
    cout << "Train: " << train_data.images.size()
         << "  Test: " << test_data.images.size() << endl;

    int input_dim = 784;
    vector<LayerGPU> layers;
    layers.reserve(3);
    layers.emplace_back(input_dim, 128);
    layers.emplace_back(128,       256);
    layers.emplace_back(256,       NUM_CLASS );

    int   epochs        = 50;
    int   batch_size    = 500;
    float learning_rate = 0.01f;
    int   total_train   = train_data.images.size();
    int   num_samples   = 50000;   // first 50k for training
    int   val_start     = 50000;   // last 10k for validation
    int   val_samples   = total_train - val_start; 

    vector<int> indices(num_samples);
    iota(indices.begin(), indices.end(), 0);
    mt19937 g(random_device{}());

    float *d_batch_x, *d_h1_z, *d_h1, *d_h2_z, *d_h2, *d_out;
    float *d_delta, *d_next_delta;
    float *d_epoch_loss;
    int   *d_labels, *d_epoch_correct;

    CUDA_CHECK(cudaMalloc(&d_batch_x,    batch_size * input_dim          * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h1_z,       batch_size * layers[0].out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h1,         batch_size * layers[0].out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h2_z,       batch_size * layers[1].out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h2,         batch_size * layers[1].out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,        batch_size * NUM_CLASS           * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta,      batch_size * layers[2].in_size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_delta, batch_size * layers[1].in_size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels,     batch_size                       * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_epoch_loss,    sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_epoch_correct, sizeof(int)));


    auto forward_layer = [&](LayerGPU& L, const float* d_in, float* d_z, int B) {
        int n = B * L.out_size;
        broadcast_bias_kernel<<<(n + blockSize-1)/blockSize, blockSize>>>(d_z, L.d_biases, B, L.out_size);
        my_gemm(cublas_handle, Trans::No, Trans::No, B, L.out_size, L.in_size,
                1.0f, d_in, L.d_weights, 1.0f, d_z);
    };

    auto update_bias = [&](float* d_b, const float* d_delta, float scale, int B, int out) {
        reduce_bias_kernel<<<(out + blockSize-1)/blockSize, blockSize>>>(
            d_delta, d_b, scale, B, out);
    };

    // ── Training loop ─────────────────────────────────────────────────────────
    auto start_train = chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(indices.begin(), indices.end(), g);

        // Zero epoch accumulators once 
        CUDA_CHECK(cudaMemset(d_epoch_loss,    0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_epoch_correct, 0, sizeof(int)));

        for (int bs = 0; bs < num_samples; bs += batch_size) {
            int be    = min(bs + batch_size, num_samples);
            int B     = be - bs;
            float scale = learning_rate / B;

            vector<float> batch_x(B * input_dim);
            vector<int>   batch_labels(B);
            for (int i = 0; i < B; i++) {
                int idx = indices[bs + i];
                copy(train_data.images[idx].begin(), train_data.images[idx].end(),
                     batch_x.begin() + i * input_dim);
                batch_labels[i] = train_data.labels[idx];
            }
            CUDA_CHECK(cudaMemcpy(d_batch_x, batch_x.data(),
                                  B * input_dim * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels, batch_labels.data(),
                                  B * sizeof(int), cudaMemcpyHostToDevice));

            // ── Forward ───────────────────────────────────────────────────────
            forward_layer(layers[0], d_batch_x, d_h1_z, B);
            relu_kernel<<<(B * layers[0].out_size + blockSize-1)/blockSize, blockSize>>>(
                d_h1, d_h1_z, B * layers[0].out_size);

            forward_layer(layers[1], d_h1, d_h2_z, B);
            relu_kernel<<<(B * layers[1].out_size + blockSize-1)/blockSize, blockSize>>>(
                d_h2, d_h2_z, B * layers[1].out_size);

            forward_layer(layers[2], d_h2, d_out, B);
            softmax_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(d_out, B, NUM_CLASS);

            cross_entropy_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(
                d_out, d_labels, B, NUM_CLASS, d_epoch_loss);
            argmax_correct_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(
                d_out, d_labels, B, NUM_CLASS, d_epoch_correct);

            // Backward
            softmax_grad_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(d_out, d_labels, B, NUM_CLASS);

            // Layer 2
            my_gemm(cublas_handle, Trans::Yes, Trans::No,
                layers[2].in_size, layers[2].out_size, B, -scale, d_h2, d_out, 1.0f, layers[2].d_weights);
            update_bias(layers[2].d_biases, d_out, scale, B, layers[2].out_size);

            my_gemm(cublas_handle, Trans::No, Trans::Yes,
                B, layers[2].in_size, layers[2].out_size, 1.0f, d_out, layers[2].d_weights, 0.0f, d_delta);
            relu_grad_kernel<<<(B * layers[2].in_size + blockSize-1)/blockSize, blockSize>>>(
                d_delta, d_h2_z, B * layers[2].in_size);

            // Layer 1
            my_gemm(cublas_handle,Trans::Yes, Trans::No,
                layers[1].in_size, layers[1].out_size, B, -scale, d_h1, d_delta, 1.0f, layers[1].d_weights);
            update_bias(layers[1].d_biases, d_delta, scale, B, layers[1].out_size);

            my_gemm(cublas_handle, Trans::No, Trans::Yes,
                B, layers[1].in_size, layers[1].out_size, 1.0f, d_delta, layers[1].d_weights, 0.0f, d_next_delta);
            relu_grad_kernel<<<(B * layers[1].in_size + blockSize-1)/blockSize, blockSize>>>(
                d_next_delta, d_h1_z, B * layers[1].in_size);

            // Layer 0
            my_gemm( cublas_handle,Trans::Yes, Trans::No,
                layers[0].in_size, layers[0].out_size, B, -scale, d_batch_x, d_next_delta, 1.0f, layers[0].d_weights);
            update_bias(layers[0].d_biases, d_next_delta, scale, B, layers[0].out_size);

        } // end batch

        float h_loss; int h_correct;
        CUDA_CHECK(cudaMemcpy(&h_loss,    d_epoch_loss,    sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_correct, d_epoch_correct, sizeof(int),   cudaMemcpyDeviceToHost));

        // ── Validation ────────────────────────────────────────────────────────
        float h_val_loss = 0.0f; int h_val_correct = 0;
        float *d_val_loss; int *d_val_correct;
        CUDA_CHECK(cudaMalloc(&d_val_loss,    sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_val_correct, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_val_loss,    0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_val_correct, 0, sizeof(int)));

        for (int bs = 0; bs < val_samples; bs += batch_size) {
            int be = min(bs + batch_size, val_samples);
            int B  = be - bs;

            vector<float> batch_x(B * input_dim);
            vector<int>   batch_labels(B);
            for (int i = 0; i < B; i++) {
                int idx = val_start + bs + i;
                copy(train_data.images[idx].begin(), train_data.images[idx].end(),
                     batch_x.begin() + i * input_dim);
                batch_labels[i] = train_data.labels[idx];
            }
            CUDA_CHECK(cudaMemcpy(d_batch_x, batch_x.data(),
                                  B * input_dim * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels, batch_labels.data(),
                                  B * sizeof(int), cudaMemcpyHostToDevice));

            forward_layer(layers[0], d_batch_x, d_h1_z, B);
            relu_kernel<<<(B * layers[0].out_size + blockSize-1)/blockSize, blockSize>>>(
                d_h1, d_h1_z, B * layers[0].out_size);
            forward_layer(layers[1], d_h1, d_h2_z, B);
            relu_kernel<<<(B * layers[1].out_size + blockSize-1)/blockSize, blockSize>>>(
                d_h2, d_h2_z, B * layers[1].out_size);
            forward_layer(layers[2], d_h2, d_out, B);
            softmax_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(d_out, B, NUM_CLASS);

            cross_entropy_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(
                d_out, d_labels, B, NUM_CLASS, d_val_loss);
            argmax_correct_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(
                d_out, d_labels, B, NUM_CLASS, d_val_correct);
        }

        CUDA_CHECK(cudaMemcpy(&h_val_loss,    d_val_loss,    sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_val_correct, d_val_correct, sizeof(int),   cudaMemcpyDeviceToHost));
        cudaFree(d_val_loss); cudaFree(d_val_correct);

        cout << "Epoch " << epoch+1
             << " | Train Loss: " << (h_loss / num_samples)
             << " | Train Acc: "  << (100.0f * h_correct / num_samples) << "%"
             << " | Val Loss: "   << (h_val_loss / val_samples)
             << " | Val Acc: "    << (100.0f * h_val_correct / val_samples) << "%"
             << endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_train = chrono::high_resolution_clock::now();

    // Inference
    auto start_infer = chrono::high_resolution_clock::now();
    int test_total = test_data.images.size();

    CUDA_CHECK(cudaMemset(d_epoch_correct, 0, sizeof(int)));

    for (int bs = 0; bs < test_total; bs += batch_size) {
        int be = min(bs + batch_size, test_total);
        int B  = be - bs;

        vector<float> batch_x(B * input_dim);
        vector<int>   batch_labels(B);
        for (int i = 0; i < B; i++) {
            int idx = bs + i;
            copy(test_data.images[idx].begin(), test_data.images[idx].end(),
                 batch_x.begin() + i * input_dim);
            batch_labels[i] = test_data.labels[idx];
        }
        CUDA_CHECK(cudaMemcpy(d_batch_x, batch_x.data(),
                              B * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_labels, batch_labels.data(),
                              B * sizeof(int), cudaMemcpyHostToDevice));

        forward_layer(layers[0], d_batch_x, d_h1_z, B);
        relu_kernel<<<(B * layers[0].out_size + blockSize-1)/blockSize, blockSize>>>(
            d_h1, d_h1_z, B * layers[0].out_size);

        forward_layer(layers[1], d_h1, d_h2_z, B);
        relu_kernel<<<(B * layers[1].out_size + blockSize-1)/blockSize, blockSize>>>(
            d_h2, d_h2_z, B * layers[1].out_size);

        forward_layer(layers[2], d_h2, d_out, B);
        softmax_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(d_out, B, NUM_CLASS);

        argmax_correct_kernel<<<(B + blockSize-1)/blockSize, blockSize>>>(
            d_out, d_labels, B, NUM_CLASS, d_epoch_correct);
    }

    int h_test_correct;
    CUDA_CHECK(cudaMemcpy(&h_test_correct, d_epoch_correct, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_infer = chrono::high_resolution_clock::now();

    cout << " | Test Acc: " << (100.0f * h_test_correct / test_total) << "%" << endl;

    double train_time = chrono::duration<double>(end_train - start_train).count();
    double infer_time = chrono::duration<double>(end_infer - start_infer).count();
    cout << "Train time: " << train_time << endl;
    cout << "Infer time: " << infer_time << endl;
    cout << "Grind Rate: " << (epochs * num_samples) / train_time << endl;




    cudaFree(d_batch_x);  cudaFree(d_h1_z);       cudaFree(d_h1);
    cudaFree(d_h2_z);     cudaFree(d_h2);          cudaFree(d_out);
    cudaFree(d_delta);    cudaFree(d_next_delta);   cudaFree(d_labels);
    cudaFree(d_epoch_loss);   cudaFree(d_epoch_correct);
#ifdef USE_CUBLAS
    cublasDestroy(cublas_handle);
#endif
    return 0;
}
