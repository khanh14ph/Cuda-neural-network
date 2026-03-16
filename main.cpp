#include <iostream>
#include <vector>

#include <cmath>
#include <algorithm>
#include <random>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include "layer.h"
#include "data.h"

using namespace std;

#define NUM_CLASS 10

void relu(vector<float>& v) {
    for (auto& x : v) x = max(0.0f, x);
}
float cross_entropy(vector<float>& predict, vector<int>& targets, int batch_size) {
    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; i++)
        total_loss -= log(predict[i * NUM_CLASS + targets[i]] + 1e-7f);
    return total_loss;
}
vector<int> arg_max(vector<float>& predict, int batch_size) {
    vector<int> result;
    for (int i = 0; i < batch_size; i++)
        result.push_back(max_element(&predict[i * NUM_CLASS], &predict[(i + 1) * NUM_CLASS]) - &predict[i * NUM_CLASS]);
    return result;
}
void softmax(vector<float>& v, int batch_size) {
    int num_class = v.size() / batch_size;
    for (int i = 0; i < (int)v.size(); i += num_class) {
        float max_val = *max_element(v.begin() + i, v.begin() + i + num_class);
        float sum = 0.0f;
        for (int j = i; j < i + num_class; j++) sum += exp(v[j] - max_val);
        for (int j = i; j < i + num_class; j++) v[j] = exp(v[j] - max_val) / sum;
    }
}


int main() {
    string base_path = "/home/khanhnd/HPC/project-3-winter-2026-khanh14ph/milestone1/data/";
    string train_img = base_path + "train-images-idx3-ubyte";
    string train_lbl = base_path + "train-labels-idx1-ubyte";
    string test_img  = base_path + "t10k-images-idx3-ubyte";
    string test_lbl  = base_path + "t10k-labels-idx1-ubyte";

    cout << "Loading MNIST Train..." << endl;
    MNISTData train_data = load_mnist(train_img, train_lbl);
    cout << "Loaded " << train_data.images.size() << " images." << endl;

    cout << "Loading MNIST Test..." << endl;
    MNISTData test_data = load_mnist(test_img, test_lbl);
    cout << "Loaded " << test_data.images.size() << " images." << endl;

    // Setup Network (784 -> 128 -> 256 -> 10)
    vector<Layer> layers;
    int input_dim = 784;
    layers.emplace_back(input_dim, 128);
    layers.emplace_back(128, 256);
    layers.emplace_back(256, 10);

    // Hyperparameters
    int epochs      = 50;
    int batch_size  = 500;
    float learning_rate = 0.01f;
    int total_train = train_data.images.size();
    int num_samples = 50000;   // first 50k for training
    int val_start   = 50000;   // last 10k for validation
    int val_samples = total_train - val_start;
    vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; i++) indices[i] = i;

    random_device rd;
    mt19937 g(rd());


    vector<float> delta(batch_size * layers[2].in_size);
    vector<float> next_delta(batch_size * layers[1].in_size);

    // Training Loop
    auto start_train = chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(indices.begin(), indices.end(), g);
        int correct = 0;
        float total_loss = 0.0f;

        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int batch_end         = min(batch_start + batch_size, num_samples);
            int current_batch_size = batch_end - batch_start;
            float scale           = learning_rate / current_batch_size;

            vector<float> batch_x(current_batch_size * input_dim);
            vector<int>   batch_labels;

            for (int i = 0; i < current_batch_size; i++) {
                int idx = indices[batch_start + i];
                copy(train_data.images[idx].begin(), train_data.images[idx].end(),
                     batch_x.begin() + i * input_dim);
                batch_labels.push_back(train_data.labels[idx]);
            }

            // Forward
            layers[0].input_cache = batch_x;
            vector<float> h1 = layers[0].forward(batch_x);
            layers[0].z_cache = h1;
            relu(h1);

            layers[1].input_cache = h1;
            vector<float> h2 = layers[1].forward(h1);
            layers[1].z_cache = h2;
            relu(h2);

            layers[2].input_cache = h2;
            vector<float> out = layers[2].forward(h2);
            softmax(out, current_batch_size);

            total_loss += cross_entropy(out, batch_labels, current_batch_size);
            vector<int> preds = arg_max(out, current_batch_size);
            for (int i = 0; i < current_batch_size; i++)
                if (preds[i] == batch_labels[i]) correct++;

            // Backward
            for (int i = 0; i < current_batch_size; i++)
                out[i * NUM_CLASS + batch_labels[i]] -= 1.0f;

            // Layer 2
            // weights -= scale * input2^T @ out_delta
            gemm(Trans::Yes, Trans::No,
                 layers[2].in_size, layers[2].out_size, current_batch_size,
                 -scale, layers[2].input_cache.data(), out.data(),
                 1.0f, layers[2].weights.data());
            // biases -= scale * sum(out_delta, axis=0)
            for (int i = 0; i < current_batch_size; i++)
                for (int j = 0; j < layers[2].out_size; j++)
                    layers[2].biases[j] -= scale * out[i * layers[2].out_size + j];
            // propagate: delta = (out_delta @ W2^T) * relu'(z1)
            gemm(Trans::No, Trans::Yes,
                 current_batch_size, layers[2].in_size, layers[2].out_size,
                 1.0f, out.data(), layers[2].weights.data(),
                 0.0f, delta.data());
            for (int i = 0; i < current_batch_size * layers[2].in_size; i++)
                delta[i] *= (layers[1].z_cache[i] > 0.0f ? 1.0f : 0.0f);

            // Layer 1
            // weights -= scale * input1^T @ delta
            gemm(Trans::Yes, Trans::No,
                 layers[1].in_size, layers[1].out_size, current_batch_size,
                 -scale, layers[1].input_cache.data(), delta.data(),
                 1.0f, layers[1].weights.data());
            for (int i = 0; i < current_batch_size; i++)
                for (int j = 0; j < layers[1].out_size; j++)
                    layers[1].biases[j] -= scale * delta[i * layers[1].out_size + j];
            // propagate: next_delta = (delta @ W1^T) * relu'(z0)
            gemm(Trans::No, Trans::Yes,
                 current_batch_size, layers[1].in_size, layers[1].out_size,
                 1.0f, delta.data(), layers[1].weights.data(),
                 0.0f, next_delta.data());
            for (int i = 0; i < current_batch_size * layers[1].in_size; i++)
                next_delta[i] *= (layers[0].z_cache[i] > 0.0f ? 1.0f : 0.0f);

            // Layer 0
            // weights -= scale * input0^T @ next_delta
            gemm(Trans::Yes, Trans::No,
                 layers[0].in_size, layers[0].out_size, current_batch_size,
                 -scale, layers[0].input_cache.data(), next_delta.data(),
                 1.0f, layers[0].weights.data());
            for (int i = 0; i < current_batch_size; i++)
                for (int j = 0; j < layers[0].out_size; j++)
                    layers[0].biases[j] -= scale * next_delta[i * layers[0].out_size + j];

        } // End Batch

        // Validation
        int val_correct = 0;
        float val_loss = 0.0f;
        for (int bs = 0; bs < val_samples; bs += batch_size) {
            int be = min(bs + batch_size, val_samples);
            int B  = be - bs;

            vector<float> batch_x(B * input_dim);
            vector<int>   batch_labels;
            for (int i = 0; i < B; i++) {
                int idx = val_start + bs + i;
                copy(train_data.images[idx].begin(), train_data.images[idx].end(),
                     batch_x.begin() + i * input_dim);
                batch_labels.push_back(train_data.labels[idx]);
            }

            vector<float> h1 = layers[0].forward(batch_x);
            relu(h1);
            vector<float> h2 = layers[1].forward(h1);
            relu(h2);
            vector<float> out = layers[2].forward(h2);
            softmax(out, B);

            val_loss += cross_entropy(out, batch_labels, B);
            vector<int> preds = arg_max(out, B);
            for (int i = 0; i < B; i++)
                if (preds[i] == batch_labels[i]) val_correct++;
        }

        cout << "Epoch " << epoch + 1
             << " | Train Loss: " << (total_loss / num_samples)
             << " | Train Acc: "  << (100.0f * correct / num_samples) << "%"
             << " | Val Loss: "   << (val_loss / val_samples)
             << " | Val Acc: "    << (100.0f * val_correct / val_samples) << "%"
             << endl;
    }
    auto end_train = chrono::high_resolution_clock::now();

    // Inference
    auto start_infer = chrono::high_resolution_clock::now();
    int test_correct = 0;
    int test_total   = test_data.images.size();

    for (int batch_start = 0; batch_start < test_total; batch_start += batch_size) {
        int batch_end          = min(batch_start + batch_size, test_total);
        int current_batch_size = batch_end - batch_start;

        vector<float> batch_x(current_batch_size * input_dim);
        vector<int>   batch_labels;

        for (int i = 0; i < current_batch_size; i++) {
            int idx = batch_start + i;
            copy(test_data.images[idx].begin(), test_data.images[idx].end(),
                 batch_x.begin() + i * input_dim);
            batch_labels.push_back(test_data.labels[idx]);
        }

        vector<float> h1 = layers[0].forward(batch_x);
        relu(h1);
        vector<float> h2 = layers[1].forward(h1);
        relu(h2);
        vector<float> out = layers[2].forward(h2);
        softmax(out, current_batch_size);

        vector<int> preds = arg_max(out, current_batch_size);
        for (int i = 0; i < current_batch_size; i++)
            if (preds[i] == batch_labels[i]) test_correct++;
    }

    cout << " | Test Acc: " << (100.0f * test_correct / test_total) << "%" << endl;
    auto end_infer = chrono::high_resolution_clock::now();

    double total_train_time = chrono::duration<double>(end_train - start_train).count();
    double total_infer_time = chrono::duration<double>(end_infer - start_infer).count();
    float  grind_rate       = (epochs * num_samples) / total_train_time;

    cout << "Train time: " << total_train_time << endl;
    cout << "Infer time: " << total_infer_time << endl;
    cout << "Grind Rate: " << grind_rate << endl;
    return 0;
}
