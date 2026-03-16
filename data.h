#include <iostream>
#include <vector>
#include <fstream>
using namespace std;
struct MNISTData {
    vector<vector<float>> images;
    vector<uint8_t> labels;
};
uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000);
}

MNISTData load_mnist(string img_file, string lbl_file) {
    MNISTData data;

    ifstream lbl_stream(lbl_file, ios::binary);
    if (!lbl_stream) { cerr << "Error opening " << lbl_file << endl; exit(1); }
    
    uint32_t magic, num_items;
    lbl_stream.read((char*)&magic, 4);
    lbl_stream.read((char*)&num_items, 4);
    num_items = swap_endian(num_items);
    
    data.labels.resize(num_items);
    lbl_stream.read((char*)data.labels.data(), num_items);


    ifstream img_stream(img_file, ios::binary);
    if (!img_stream) { cerr << "Error opening " << img_file << endl; exit(1); }
    
    uint32_t rows, cols;
    img_stream.read((char*)&magic, 4);
    img_stream.read((char*)&num_items, 4);
    img_stream.read((char*)&rows, 4);
    img_stream.read((char*)&cols, 4);
    
    rows = swap_endian(rows);
    cols = swap_endian(cols);
    int pixels_per_img = rows * cols;

    data.images.resize(data.labels.size(), vector<float>(pixels_per_img));
    
    vector<uint8_t> buffer(pixels_per_img);
    for (size_t i = 0; i < data.labels.size(); i++) {
        img_stream.read((char*)buffer.data(), pixels_per_img);
        for (int p = 0; p < pixels_per_img; p++) {
            data.images[i][p] = buffer[p] / 255.0f;
        }
    }
    return data;
}