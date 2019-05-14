#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

class GraphAlexNetExample : public Example
{
public:
    GraphAlexNetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "AlexNet")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

         // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))

              << ConvolutionLayer(
                    11U, 11U, 96U,
                    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy", weights_layout), 
                    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"), 
                    PadStrideInfo(4, 4, 0, 0))
                    .set_name(conv1)
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 9.999999747378752e-05, 0.75f))).set_name(norm1)
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)).set_name(pool1)
              << ConvolutionLayer(5U, 5U, 256U, get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_w.npy", weights_layout), get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_b.npy"), PadStrideInfo(1, 1, 2, 2)).set_name(conv2)
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 9.999999747378752e-05, 0.75f))).set_name(norm2)
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)).set_name(pool2)
              << ConvolutionLayer(3U, 3U, 384U, get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy", weights_layout), get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"), PadStrideInfo(1, 1, 1, 1)).set_name(conv3)
              << ConvolutionLayer(3U, 3U, 384U, get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy", weights_layout), get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"), PadStrideInfo(1, 1, 1, 1)).set_name(conv4)
              << ConvolutionLayer(3U, 3U, 256U, get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy", weights_layout), get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"), PadStrideInfo(1, 1, 1, 1)).set_name(conv5)
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)).set_name(pool5)
              << FullyConnectedLayer(4096U, get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy", weights_layout), get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy")).set_name(fc6)
              << FullyConnectedLayer(4096U, get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy", weights_layout), get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy")).set_name(fc7)
              << FullyConnectedLayer(1000U, get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy", weights_layout), get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy")).set_name(fc8)
              << SoftmaxLayer().set_name(prob) 
                



        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_file  = common_params.tuner_file;

        graph.finalize(common_params.target, config);

        return true;
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;
};

int main(int argc, char **argv) {


  arm_compute::CLScheduler::get().default_init();

  unsigned times[NO_OF_RUNS];

  auto t1 = std::chrono::high_resolution_clock::now();

  for (unsigned i = 0; i < NO_OF_RUNS; i++) {

    auto t2 = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    t1 = t2;
  }

  for (unsigned i = 0; i < NO_OF_RUNS; i++) {
    std::cout << times[i] << std::endl;
  }
  
  return 0;
}
  