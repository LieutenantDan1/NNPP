#include <string>
#include <vector>
#include <array>
#include <limits>
#include <filesystem>

static_assert(CHAR_BIT == 8, "Type char must be 8 bits wide.");
static_assert(sizeof(float) == 4, "Type float must be 4 bytes wide.");
static_assert(std::numeric_limits<float>::is_iec559, "Only IEEE-754 floating point types are supported.");

typedef std::vector<std::vector<float>> matrix_2d;
typedef std::vector<matrix_2d> matrix_3d;

namespace nn
{
	namespace func
	{
		float sigmoid(float x)
		{
			return 1.0f / (1.0f + expf(-x));
		}
		float d_sigmoid(float x)
		{
			float y = 1.0f / (1.0f + expf(-x));
			return y - y * y;
		}
		float relu(float x)
		{
			return x < 0.0f ? 0.0f : x;
		}
		float d_relu(float x)
		{
			return x < 0.0f ? 0.0f : 1.0f;
		}
		float lrelu(float x)
		{
			return x <= 0.0f ? 0.01f * x : x;
		}
		float d_lrelu(float x)
		{
			return x <= 0.0f ? 0.01f : 1.0f;
		}
		float swish(float x)
		{
			return x / (1.0f + expf(-x));
		}
		float d_swish(float x)
		{
			float s = 1.0f / (1.0f + expf(-x));
			float y = x * s;
			return y + s * (1.0f - y);
		}
	}

	class neural_network
	{
	private:
		template <typename T, bool big_endian>
		static void write(std::ofstream& out_file, T value)
		{
			char* b = reinterpret_cast<char*>(&value);
			const bool format = std::endian::native == std::endian::little;
			if constexpr (format != big_endian)
			{
				out_file.write(b, sizeof(T));
			}
			else
			{
				for (size_t i = sizeof(T); i > 0; )
				{
					out_file.write(&b[--i], 1);
				}
			}
		}

		template <typename T, bool big_endian>
		static T read(std::ifstream& in_file)
		{
			T value;
			char* b = reinterpret_cast<char*>(&value);
			const bool format = std::endian::native == std::endian::little;
			if constexpr (format != big_endian)
			{
				in_file.read((char*)&value, sizeof(T));
			}
			else
			{
				for (size_t i = sizeof(T); i > 0; )
				{
					in_file.read(&b[--i], 1);
				}
			}
			return value;
		}

		static bool test_str(const char* str, std::ifstream& file)
		{
			for (size_t i = 0; str[i] != '\0'; i++)
			{
				if (read<char, false>(file) != str[i]) return false;
			}
			return true;
		}

		size_t num_layers = 0;
		float(*activation_func)(float) = nullptr;
		float(*d_activation_func)(float) = nullptr;
		float learning_rate = 0.04f;
		matrix_2d activations;
		matrix_2d weighted_sums;
		matrix_2d biases;
		matrix_2d bias_updates;
		matrix_3d weights;
		matrix_3d weight_updates;
		matrix_2d errors;
		size_t num_pending_backprops = 0;
	public:
		neural_network();
		neural_network
		(
			float(*activation_func)(float),
			float(*d_activation_func)(float)
		): 
		activation_func(activation_func), 
		d_activation_func(d_activation_func) 
		{}
		neural_network
		(
			size_t num_layers,
			std::initializer_list<size_t> layer_sizes,
			float learning_rate, 
			float(*activation_func)(float),
			float(*d_activation_func)(float)
		):
		num_layers(num_layers),
		learning_rate(learning_rate), 
		activation_func(activation_func), 
		d_activation_func(d_activation_func)
		{
			if (num_layers == 0)
			{
				throw std::invalid_argument("Cannot create neural network with 0 layers.");
			}
			if (layer_sizes.size() != num_layers)
			{
				throw std::invalid_argument("Number of provided layer sizes does not match number of layers.");
			}
			weighted_sums = matrix_2d(num_layers);
			activations = matrix_2d(num_layers);
			biases = matrix_2d(num_layers);
			bias_updates = matrix_2d(num_layers);
			weights = matrix_3d(num_layers);
			weight_updates = matrix_3d(num_layers);
			errors = matrix_2d(num_layers);
			auto it = layer_sizes.begin();
			for (size_t i = 0; i < num_layers; i++)
			{
				size_t layer_size = *it;
				if (layer_size == 0)
				{
					throw std::invalid_argument("Cannot create layer of size 0.");
				}
				weighted_sums[i] = std::vector<float>(layer_size);
				activations[i] = std::vector<float>(layer_size);
				if (i > 0)
				{
					biases[i] = std::vector<float>(layer_size);
					bias_updates[i] = std::vector<float>(layer_size);
					weights[i] = matrix_2d(layer_size);
					weight_updates[i] = matrix_2d(layer_size);
					errors[i] = std::vector<float>(layer_size);
					size_t prev_layer_size = activations[i - 1].size();
					for (size_t j = 0; j < layer_size; j++)
					{
						biases[i][j] = 0.0f;
						weights[i][j] = std::vector<float>(prev_layer_size);
						weight_updates[i][j] = std::vector<float>(prev_layer_size);
						for (size_t k = 0; k < prev_layer_size; k++)
						{
							weights[i][j][k] = 0.0f;
						}
					}
				}
				it = std::next(it);
			}
		}
		void randomize(float min_weight, float max_weight, float min_bias, float max_bias, uint_fast64_t seed)
		{
			if (num_layers == 0)
			{
				throw std::runtime_error("Operation attempt on uninitialized neural network.");
			}
			std::mt19937_64 rng(seed);
			std::uniform_real_distribution<float> distr_weight(min_weight, max_weight);
			std::uniform_real_distribution<float> distr_bias(min_bias, max_bias);
			for (size_t i = 1; i < num_layers; i++)
			{
				size_t layer_size = activations[i].size();
				size_t prev_layer_size = activations[i - 1].size();
				for (size_t j = 0; j < layer_size; j++)
				{
					biases[i][j] += distr_bias(rng);
					for (size_t k = 0; k < prev_layer_size; k++)
					{
						weights[i][j][k] += distr_weight(rng);
					}
				}
			}
		}
		size_t get_num_layers() const
		{
			return num_layers;
		}
		size_t get_layer_size(size_t layer, bool reverse = false) const
		{
			if (num_layers == 0)
			{
				throw std::runtime_error("Operation attempt on uninitialized neural network.");
			}
			if (reverse)
			{
				layer = activations.size() - layer - 1;
			}
			if (layer >= activations.size())
			{
				throw std::out_of_range("Layer does not exist.");
			}
			return activations[layer].size();
		}
		size_t get_num_pending_backprops() const
		{
			return num_pending_backprops;
		}
		void set_input(std::vector<float>& source)
		{
			size_t size = std::min(activations[0].size(), source.size());
			for (size_t i = 0; i < size; i++)
			{
				activations[0][i] = source[i];
			}
		}
		void get_output(std::vector<float>& destination)
		{
			destination = activations.back();
		}
		void save(std::string filename)
		{
			if (num_layers == 0)
			{
				throw std::runtime_error("Operation attempt on uninitialized neural network.");
			}
			std::ofstream out_file(filename, std::ios::binary);
			out_file << "NNSF";
			write<uint64_t, false>(out_file, num_layers);
			for (size_t i = 0; i < num_layers; i++)
			{
				size_t layer_size = activations[i].size();
				out_file << "layer";
				write<uint64_t, false>(out_file, layer_size);
				if (i > 0)
				{
					size_t prev_layer_size = activations[i - 1].size();
					for (size_t j = 0; j < layer_size; j++)
					{
						write<float, false>(out_file, biases[i][j]);
						for (size_t k = 0; k < prev_layer_size; k++)
						{
							write<float, false>(out_file, weights[i][j][k]);
						}
					}
				}
			}
			out_file.close();
		}
		bool load(std::string filename)
		{
			if (!std::filesystem::exists(filename)) 
			{
				return false;
			}
			std::ifstream in_file(filename, std::ios::binary);
			if (!in_file) 
			{
				return false;
			}
			if (!test_str("NNSF", in_file)) 
			{
				in_file.close();
				return false;
			}
			num_layers = read<uint64_t, false>(in_file);
			try
			{
				weighted_sums = matrix_2d(num_layers);
				activations = matrix_2d(num_layers);
				biases = matrix_2d(num_layers);
				bias_updates = matrix_2d(num_layers);
				weights = matrix_3d(num_layers);
				weight_updates = matrix_3d(num_layers);
				errors = matrix_2d(num_layers);
			}
			catch (std::bad_alloc& e)
			{
				num_layers = 0;
				in_file.close();
				return false;
			}
			for (size_t i = 0; i < num_layers; i++)
			{
				if (!test_str("layer", in_file)) 
				{
					num_layers = 0;
					in_file.close();
					return false;
				}
				size_t layer_size = read<uint64_t, false>(in_file);
				weighted_sums[i] = std::vector<float>(layer_size);
				activations[i] = std::vector<float>(layer_size);
				if (i != 0)
				{
					biases[i] = std::vector<float>(layer_size);
					bias_updates[i] = std::vector<float>(layer_size);
					weights[i] = matrix_2d(layer_size);
					weight_updates[i] = matrix_2d(layer_size);
					errors[i] = std::vector<float>(layer_size);
					size_t prev_layer_size = activations[i - 1].size();
					for (size_t j = 0; j < layer_size; j++)
					{
						biases[i][j] = read<float, false>(in_file);
						weights[i][j] = std::vector<float>(prev_layer_size);
						for (size_t k = 0; k < prev_layer_size; k++)
						{
							weights[i][j][k] = read<float, false>(in_file);
						}
					}
				}
			}
			if (in_file.fail())
			{
				num_layers = 0;
				in_file.close();
				return false;
			}
			in_file.close();
			return true;
		}
		void propagate()
		{
			if (num_layers == 0)
			{
				throw std::runtime_error("Operation attempt on uninitialized neural network.");
			}
			if (activation_func == nullptr || d_activation_func == nullptr)
			{
				throw std::runtime_error("Undefined activation function(s).");
			}
			for (size_t i = 1; i < num_layers; i++)
			{
				size_t layer_size = activations[i].size();
				size_t prev_i = i - 1;
				size_t prev_layer_size = activations[i - 1].size();
				for (size_t j = 0; j < layer_size; j++)
				{
					errors[i][j] = 0.0f;
					float sum = biases[i][j];
					for (size_t k = 0; k < prev_layer_size; k++)
					{
						sum += weights[i][j][k] * activations[prev_i][k];
					}
					weighted_sums[i][j] = sum;
					activations[i][j] = activation_func(sum);
				}
			}
		}
		float propagate_backward(std::vector<float>& expected)
		{
			propagate();
			float cost = 0.0f;
			size_t back_layer = num_layers - 1;
			size_t back_layer_size = activations[back_layer].size();
			for (size_t i = 0; i < back_layer_size; i++)
			{
				float error = activations[back_layer][i] - expected[i];
				errors[back_layer][i] = error;
				cost += error * error;
			}
			for (size_t i = num_layers - 2; ~i; i--)
			{
				size_t prev_i = i + 1;
				size_t layer_size = activations[i].size();
				size_t prev_layer_size = activations[prev_i].size();
				for (size_t k = 0; k < prev_layer_size; k++)
				{
					float part_deriv = d_activation_func(activations[prev_i][k]) * 2.0f * errors[prev_i][k];
					bias_updates[prev_i][k] -= learning_rate * part_deriv;
					for (size_t j = 0; j < layer_size; j++)
					{
						if (i > 0) 
						{
							errors[i][j] += weights[prev_i][k][j] * part_deriv;
						}
						weight_updates[prev_i][k][j] -= learning_rate * activations[i][j] * part_deriv;
					}
				}
			}
			num_pending_backprops++;
			return cost;
		}
		void apply_backprops()
		{
			if (num_layers == 0)
			{
				throw std::runtime_error("Operation attempt on uninitialized neural network.");
			}
			if (num_pending_backprops != 0)
			{
				for (size_t i = 1; i < num_layers; i++)
				{
					size_t layer_size = activations[i].size();
					size_t prev_layer_size = activations[i - 1].size();
					for (size_t j = 0; j < layer_size; j++)
					{
						errors[i][j] = 0.0f;
						biases[i][j] += bias_updates[i][j] / num_pending_backprops;
						bias_updates[i][j] = 0.0f;
						for (size_t k = 0; k < prev_layer_size; k++)
						{
							weights[i][j][k] += weight_updates[i][j][k] / num_pending_backprops;
							weight_updates[i][j][k] = 0.0f;
						}
					}
				}
				num_pending_backprops = 0;
			}
		}
	};
}