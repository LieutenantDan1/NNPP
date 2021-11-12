#include <filesystem>
#include <fstream>
#include <string>
#include "nnpp.h"

float nn::relu(float x)
{
    if (x < 0.0f) return 0.01f * x;
    return x;
}

float nn::d_relu(float y)
{
    if (y > 0.0f) return 1.0f;
    return 0.01f;
}

float nn::get_random_float(float min, float max)
{
    if (min == max) return min;
    return min + (max - min) * ((float)rand() / 2147483647.0f);
}

void write_uint64(uint64_t value, std::ofstream& file, uint8_t* buffer, uint16_t* checksum = nullptr)
{
    for (uint8_t i = 0; i < 8; i++)
    {
        buffer[i] = value >> (56 - 8 * i);
        if (checksum != nullptr) *checksum += buffer[i];
    }
    file.write((char*)buffer, 8);
}
void write_uint16(uint16_t value, std::ofstream& file, uint8_t* buffer, uint16_t* checksum = nullptr)
{
    for (uint8_t i = 0; i < 2; i++)
    {
        buffer[i] = value >> (8 - 8 * i);
        if (checksum != nullptr) *checksum += buffer[i];
    }
    file.write((char*)buffer, 2);
}
void write_float32(float value, std::ofstream& file, uint8_t* buffer, uint16_t* checksum = nullptr)
{
    uint32_t temp = *(uint32_t*)&value;
    for (uint8_t i = 0; i < 4; i++)
    {
        buffer[i] = temp >> (24 - 8 * i);
        if (checksum != nullptr) *checksum += buffer[i];
    }
    file.write((char*)buffer, 4);
}
uint64_t read_uint64(std::ifstream& file, uint8_t* buffer, uint8_t& result, uint16_t* checksum = nullptr)
{
    uint64_t output = 0;
    file.read((char*)buffer, 8);
    if (file.eof()) { result = 2; return 0; }
    for (uint8_t i = 0; i < 8; i++)
    {
        output |= buffer[i] << (56 - 8 * i);
        if (checksum != nullptr) *checksum += buffer[i];
    }
    result = 0;
    return output;
}
uint16_t read_uint16(std::ifstream& file, uint8_t* buffer, uint8_t& result, uint16_t* checksum = nullptr)
{
    uint16_t output = 0;
    file.read((char*)buffer, 2);
    if (file.eof()) { result = 2; return 0; }
    for (uint8_t i = 0; i < 2; i++)
    {
        output |= buffer[i] << (8 - 8 * i);
        if (checksum != nullptr) *checksum += buffer[i];
    }
    result = 0;
    return output;
}
float read_float32(std::ifstream& file, uint8_t* buffer, uint8_t& result, uint16_t* checksum = nullptr)
{
    uint32_t output = 0;
    file.read((char*)buffer, 4);
    if (file.eof()) { result = 2; return 0.0f; }
    for (uint8_t i = 0; i < 4; i++)
    {
        output |= buffer[i] << (24 - 8 * i);
        if (checksum != nullptr) *checksum += buffer[i];
    }
    result = 0;
    return *(float*)&output;
}
uint8_t read_cstr(const char* str, std::ifstream& file, size_t len, uint8_t* buffer)
{
    file.read((char*)buffer, len);
    if (file.eof()) return 2;
    for (size_t i = 0; i < len; i++)
    {
        if ((char)buffer[i] != str[i]) return 1;
    }
    return 0;
}
nn::Neuron::Neuron() :
    value(0.0f), bias(0.0f), weights(nullptr), sum_delta_bias(0.0f),
    error(0.0f), sum_delta_weights(nullptr), prev_layer(nullptr),
    prev_layer_size(0)
{}
void nn::Neuron::shallow_copy(const Neuron& original)
{
    value = original.value;
    if (prev_layer != nullptr)
    {
        bias = original.bias;
        for (uint64_t i = 0; i < prev_layer_size; i++)
        {
            weights[i] = original.weights[i];
        }
    }
}
void nn::Neuron::initialize(Neuron* _prev_layer, size_t& _prev_layer_size, float& min_weight, float& max_weight, float& min_bias, float& max_bias)
{
    prev_layer = _prev_layer;
    prev_layer_size = _prev_layer_size;
    bias = get_random_float(min_bias, max_bias);
    weights = new float[prev_layer_size];
    sum_delta_weights = new float[prev_layer_size];
    for (uint64_t i = 0; i < prev_layer_size; i++)
    {
        weights[i] = get_random_float(min_weight, max_weight);
    }
}
void nn::Neuron::load(Neuron* _prev_layer, size_t& _prev_layer_size)
{
    prev_layer = _prev_layer;
    prev_layer_size = _prev_layer_size;
    weights = new float[prev_layer_size];
}
nn::Neuron::~Neuron()
{
    if (weights != nullptr)
    {
        delete[] weights;
        weights = nullptr;
    }
}
void nn::Neuron::update()
{
    float sum = 0.0f;
    error = 0.0f;
    for (uint64_t i = 0; i < prev_layer_size; i++)
    {
        sum += weights[i] * prev_layer[i].value;
    }
    sum += bias;
    value = relu(sum);
}
void nn::Neuron::apply_training(uint64_t& num_cases)
{
    float scale = 1.0f / num_cases;
    for (uint64_t i = 0; i < prev_layer_size; i++)
    {
        weights[i] += sum_delta_weights[i] * scale;
        sum_delta_weights[i] = 0.0f;
    }
    bias += sum_delta_bias * scale;
    sum_delta_bias = 0.0f;
}

nn::NeuralNetwork::NeuralNetwork() :
    layers(nullptr), input_layer(nullptr), output_layer(nullptr), 
    num_layers(0), layer_sizes(nullptr), num_train_cases(0)
{}
void nn::NeuralNetwork::shallow_copy(const NeuralNetwork& original)
{
    for (uint64_t i = 1; i < num_layers; i++)
    {
        Neuron* layer = layers[i];
        size_t layer_size = layer_sizes[i];
        Neuron* original_layer = original.layers[i];
        for (uint64_t j = 0; j < layer_size; j++)
        {
            layer[j].shallow_copy(original_layer[j]);
        }
    }
}
void nn::NeuralNetwork::initialize(size_t _num_layers, size_t* _layer_sizes, float _learning_rate, float min_weight, float max_weight, float min_bias, float max_bias)
{
    num_layers = _num_layers;
    layer_sizes = _layer_sizes;
    learning_rate = _learning_rate;
    layers = new Neuron*[num_layers];
    size_t prev_layer_size = layer_sizes[0];
    Neuron* prev_layer = new Neuron[prev_layer_size];
    layers[0] = prev_layer;
    for (uint64_t i = 1; i < num_layers; i++)
    {
        size_t layer_size = layer_sizes[i];
        Neuron* layer = new Neuron[layer_size];
        layers[i] = layer;
        for (uint64_t j = 0; j < layer_size; j++)
        {
            layer[j].initialize(prev_layer, prev_layer_size, min_weight, max_weight, min_bias, max_bias);
        }
        prev_layer = layer;
        prev_layer_size = layer_size;
    }
    input_layer = layers[0];
    output_layer = layers[num_layers - 1];
}
void nn::NeuralNetwork::save(std::string filename)
{
    size_t out_file_size = 
        4 + // NNSF header
        8 + // num_layers, uint64
        num_layers * 8 + // layer_size, uint64
        7; // first layer: LAYER header + checksum
    for (uint64_t i = 1; i < num_layers; i++)
    {
        out_file_size += 
            5 + // LAYER header
            layer_sizes[i - 1] * layer_sizes[i] * 4 + // weights, float32
            layer_sizes[i] * 4 + // biases, float32
            2; // checksum, uint16
    }
    std::ofstream out_file(filename, std::ios::binary);
    uint8_t* write_buffer = new uint8_t[8]; // Using one buffer for writing to avoid allocating each time
    out_file << "NNSF";
    write_uint64(num_layers, out_file, write_buffer);
    uint64_t prev_layer_size = 0;
    for (uint64_t i = 0; i < num_layers; i++)
    {
        Neuron* layer = layers[i];
        uint64_t layer_size = layer_sizes[i];
        uint16_t checksum = 0;
        out_file << "LAYER";
        write_uint64(layer_size, out_file, write_buffer, &checksum);
        if (i > 0)
        {
            for (uint64_t j = 0; j < layer_size; j++)
            {
                write_float32(layer[j].bias, out_file, write_buffer, &checksum);
                for (uint64_t k = 0; k < prev_layer_size; k++)
                {
                    write_float32(layer[j].weights[k], out_file, write_buffer, &checksum);
                }
            }
        }
        write_uint16(checksum, out_file, write_buffer);
        prev_layer_size = layer_size;
    }
    delete[] write_buffer;
}
enum LoadResult : uint8_t
{
    SUCCESS,
    FILE_DNE,
    OPEN_FAIL,
    PREM_EOF,
    BAD_HEADER,
    BAD_CHECKSUM,
};
uint8_t nn::NeuralNetwork::load(std::string filename)
{
    if (!std::filesystem::exists(filename)) return LoadResult::FILE_DNE;
    std::ifstream in_file(filename, std::ios::binary);
    if (!in_file) return LoadResult::OPEN_FAIL;
    uint8_t read_result = 0; // 0: Successful read, 1: Unexpected output, 2: End of file
    uint8_t* read_buffer = new uint8_t[8]; // Using one buffer for reading to avoid allocating each time
    read_result = read_cstr("NNSF", in_file, 4, read_buffer);
    if (read_result == 2) { delete[] read_buffer; return LoadResult::PREM_EOF; }
    if (read_result == 1) { delete[] read_buffer; return LoadResult::BAD_HEADER; }
    num_layers = read_uint64(in_file, read_buffer, read_result);
    if (read_result == 2) { delete[] read_buffer; return LoadResult::PREM_EOF; }
    layers = new Neuron*[num_layers];
    layer_sizes = new size_t[num_layers];
    uint64_t layer_idx = 0;
    Neuron* prev_layer = nullptr;
    uint64_t prev_layer_size = 0;
    for (uint64_t i = 0; i < num_layers; i++)
    {
        read_result = read_cstr("LAYER", in_file, 5, read_buffer);
        if (read_result == 2) { delete[] read_buffer; return LoadResult::PREM_EOF; }
        if (read_result == 1) { delete[] read_buffer; return LoadResult::BAD_HEADER; }
        uint16_t checksum = 0;
        uint64_t layer_size = read_uint64(in_file, read_buffer, read_result, &checksum);
        if (read_result == 2) { delete[] read_buffer; return LoadResult::PREM_EOF; }
        Neuron* layer = new Neuron[layer_size];
        layer_sizes[i] = layer_size;
        layers[i] = layer;
        if (i > 0)
        {
            for (uint64_t j = 0; j < layer_size; j++)
            {
                layer[j].load(prev_layer, prev_layer_size);
                layer[j].bias = read_float32(in_file, read_buffer, read_result, &checksum);
                if (read_result == 2) { delete[] read_buffer; return LoadResult::PREM_EOF; }
                for (uint64_t k = 0; k < prev_layer_size; k++)
                {
                    layer[j].weights[k] = read_float32(in_file, read_buffer, read_result, &checksum);
                    if (read_result == 2) { delete[] read_buffer; return LoadResult::PREM_EOF; }
                }
            }
        }
        uint16_t test_checksum = read_uint16(in_file, read_buffer, read_result);
        if (read_result == 2) { delete[] read_buffer; return LoadResult::PREM_EOF; }
        if (checksum != test_checksum) { delete[] read_buffer; return LoadResult::BAD_CHECKSUM; }
        prev_layer = layer;
        prev_layer_size = layer_size;
    }
    delete[] read_buffer;
    return LoadResult::SUCCESS;
}
nn::NeuralNetwork::~NeuralNetwork()
{
    for (uint64_t i = 1; i < num_layers; i++)
    {
        if (layers[i] != nullptr)
        {
            delete[] layers[i];
            layers[i] = nullptr;
        }
    }
    if (layers != nullptr)
    {
        delete[] layers;
        layers = nullptr;
    }
}
float* nn::NeuralNetwork::update(float* input)
{
    Neuron* first_layer = layers[0];
    uint64_t first_layer_size = layer_sizes[0];
    for (uint64_t i = 0; i < first_layer_size; i++)
    {
        first_layer[i].value = input[i];
    }
    for (uint64_t i = 1; i < num_layers; i++)
    {
        size_t layer_size = layer_sizes[i];
        Neuron* layer = layers[i];
        for (uint64_t j = 0; j < layer_size; j++)
        {
            layer[j].update();
        }
    }
    Neuron* last_layer = layers[num_layers - 1];
    uint64_t last_layer_size = layer_sizes[num_layers - 1];
    float* output = new float[last_layer_size];
    for (uint64_t i = 0; i < last_layer_size; i++)
    {
        output[i] = last_layer[i].value;
    }
    return output;
}
float nn::NeuralNetwork::train(float* input, float* expected)
{
    float* output = update(input);
    Neuron* prev_layer = layers[num_layers - 1];
    size_t prev_layer_size = layer_sizes[num_layers - 1];
    float loss = 0.0f;
    for (uint64_t i = 0; i < prev_layer_size; i++)
    {
        float error = output[i] - expected[i];
        prev_layer[i].error = error;
        loss += error * error;
    }
    delete[] output;
    for (uint64_t i = num_layers - 2; ;)
    {
        Neuron* layer = layers[i];
        size_t layer_size = layer_sizes[i];
        for (uint64_t k = 0; k < prev_layer_size; k++)
        {
            float prev_error = prev_layer[k].error;
            float partial_delta = d_relu(prev_layer[k].value) * 2.0f * prev_error;
            prev_layer[k].sum_delta_bias -= learning_rate * partial_delta;
            for (uint64_t j = 0; j < layer_size; j++)
            {
                layer[j].error += prev_layer[k].weights[j] * partial_delta;
                prev_layer[k].sum_delta_weights[j] -= learning_rate * layer[j].value * partial_delta;
            }
        }
        prev_layer = layer;
        prev_layer_size = layer_size;
        if (i == 0) break;
        else i--;
    }
    num_train_cases++;
    return loss;
}
void nn::NeuralNetwork::apply_training()
{
    if (num_train_cases != 0)
    {
        for (uint64_t i = 0; i < num_layers; i++)
        {
            Neuron* layer = layers[i];
            size_t layer_size = layer_sizes[i];
            for (uint64_t j = 0; j < layer_size; j++)
            {
                layer[j].apply_training(num_train_cases);
            }
        }
        num_train_cases = 0;
    }
}