/*
How to use the NN++ library:
    1. Create a NeuralNetwork object <network>.
    
    2. Call network.initialize().

        1. Specify number of layers (>= 2).
        2. Specify layer sizes for each layer (new size_t[] {#, #, ...}).
            The first and last numbers inside the size_t array correspond to
            the input/output layer sizes respectively.

        3. Specify learning rate (0 to 1) (default 0.1).
            If the network outputs NaNs, doesn't decrease loss or otherwise
            performs strangely during training, try lowering the learning rate.

        4. Specify min/max weight, min/max bias (default 0, 1 for both).
    
    3. Train the network
    
        1. Pass the input and expected output arrays into network.train().
            This function returns the loss for the supplied training example.

        2. Repeat step 1 many times with different training examples.
        3. Call network.apply_training() (IMPORTANT).
        4. Repeat until desired effect is achieved (e.g. loss is low enough).
    
    4. Save the network

        1. Call network.save() and specify filename (std::string).
        2. You are now CEO of Apple.

    5. Load the network

        1. Create a NeuralNetwork object <network>.
        2. Call network.load() and specify filename (std::string).
            This function returns true if the network loaded successfully,
            otherwise false.

    6. Use the network

        1. Pass your input float array into network.update()
            This function returns a float pointer to the output array, same
            size as the output layer.
*/

/*
NN++ Example Program
This neural network learns how to swap two floats, with a redundant hidden layer
for testing purposes.
Run ./example -c to train and save the neural network.
Run ./example -t to load and test the neural network.
*/

#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include "nnpp.h"

int main(int argc, char* argv[])
{
    if (argc == 2)
    {
        if (!std::strcmp(argv[1], "-c"))
        {
            srand(time(NULL));
            nn::NeuralNetwork network;
            network.initialize(3, new size_t[] {2, 16, 2}, 0.04f);
            float* input = new float[2];
            float* expected = new float[2];
            for (uint32_t g = 0; g < 20000; g++)
            {
                float average_loss = 0.0f;
                for (uint32_t i = 0; i < 1000; i++)
                {
                    input[0] = nn::get_random_float();
                    input[1] = nn::get_random_float();
                    expected[0] = input[1];
                    expected[1] = input[0];
                    average_loss += network.train(input, expected);
                }
                average_loss /= 1000.0f;
                network.apply_training();
                std::cout << "\033[2J\033[1;1H";
                std::cout << "Generation: " << g << ", Average Loss: " << average_loss << "\n";
            }
            delete[] input;
            delete[] expected;
            network.save("example_nn.nns");
        }
        else if (!std::strcmp(argv[1], "-t"))
        {
            nn::NeuralNetwork network;
            uint8_t result = network.load("example_nn.nns");
            if (result != 0)
            {
                std::cout << "Failed to load example_nn.nns. Error code: " << (uint16_t)result << "\n";
            }
            float* input = new float[] { 0.15f, 0.85f };
            float* output = network.update(input);
            std::cout << "Input: " << input[0] << ", " << input[1] << "\n";
            std::cout << "Output: " << output[0] << ", " << output[1] << "\n";
            delete[] input;
            delete[] output;
        }
    }
    
}