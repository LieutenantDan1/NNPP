namespace nn
{
	float relu(float x);
    float d_relu(float x);
    float get_random_float(float min = 0.0f, float max = 1.0f);

	class Neuron
	{
	private:
		Neuron* prev_layer;
		size_t prev_layer_size;
	public:
		float value;
		float bias;
		float* weights;
		float sum_delta_bias;
		float* sum_delta_weights;
		float error;
		Neuron();
		void shallow_copy(const Neuron& original);
		void initialize(Neuron* _prev_layer, size_t& _prev_layer_size, float& min_weight, float& max_weight, float& min_bias, float& max_bias);
		void load(Neuron* _prev_layer, size_t& _prev_layer_size);
		~Neuron();
		void update();
		void apply_training(uint64_t& num_cases);
	};
	class NeuralNetwork
	{
	private:
		size_t* layer_sizes;
		uint64_t num_train_cases;
	public:
		Neuron** layers;
		Neuron* input_layer;
		Neuron* output_layer;
		size_t num_layers;
		float learning_rate;
		NeuralNetwork();
		void shallow_copy(const NeuralNetwork& original);
		void initialize(size_t _num_layers, size_t* _layer_sizes, float _learning_rate = 0.1f, float min_weight = 0.0f, float max_weight = 1.0f, float min_bias = 0.0f, float max_bias = 1.0f);
		void save(std::string filename);
		uint8_t load(std::string filename);
		~NeuralNetwork();
		float* update(float* input);
		float train(float* input, float* expected);
		void apply_training();
	};
}
