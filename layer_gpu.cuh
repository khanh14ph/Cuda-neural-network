#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include "kernels.cuh"

struct LayerGPU {
    int in_size, out_size;
    float* d_weights;  
    float* d_biases;   
    LayerGPU(int in, int out) : in_size(in), out_size(out) {
        cudaMalloc(&d_weights, in * out * sizeof(float));
        cudaMalloc(&d_biases,  out      * sizeof(float));

        float std_dev = std::sqrt(2.0f / in);
        he_init_kernel<<<(in * out + 255) / 256, 256>>>(
            d_weights, in * out, std_dev, (unsigned long long)clock());

        cudaMemset(d_biases, 0, out * sizeof(float));
    }

    ~LayerGPU() { cudaFree(d_weights); cudaFree(d_biases); }

    LayerGPU(LayerGPU&& o) noexcept
        : in_size(o.in_size), out_size(o.out_size),
          d_weights(o.d_weights), d_biases(o.d_biases)
    { o.d_weights = o.d_biases = nullptr; }
};
