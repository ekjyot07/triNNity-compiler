preamble = '''#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

class Graph{network}Example : public Example
{
public:
    Graph{network}Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "{network}")
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
        const std::array<float, 3> mean_rgb{ { {mean_r}, {mean_g}, {mean_b} } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

         // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape({W}, {H}, {C}, {N}), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
'''

postamble = '''
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
'''

main_code_preamble_A = '''  unsigned times[NO_OF_RUNS];

  auto t1 = std::chrono::high_resolution_clock::now();

  for (unsigned i = 0; i < NO_OF_RUNS; i++) {
'''

main_code_preamble_B = '''
    auto t2 = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    t1 = t2;
  }

  for (unsigned i = 0; i < NO_OF_RUNS; i++) {
    std::cout << times[i] << std::endl;
  }
  '''

main_code_postamble = '''
  return 0;
}
  '''

class ARMCLRuntime(object):

    def __init__(self, output):
        self.output = output

    def generate(self, code, transformer):
      (in_N, in_C, in_H, in_W) = transformer.graph.get_node('data').output_shape
      self.output.write(preamble.format(network=transformer.graph.name,
                                    mean_r="122.68f", mean_g="116.67f", mean_b="104.01f",
                                    W="{}U".format(in_W), H="{}U".format(in_H),
                                    C="{}U".format(in_C), N="{}U".format(in_N)))
      self.output.write('\n')
      self.output.write(code[0])
      self.output.write('\n')
      self.output.write(postamble)
      self.output.write('\n')
      self.output.write('int main(int argc, char **argv) {\n')
      self.output.write('\n\n')
      self.output.write('arm_compute::CLScheduler::get().default_init();\n\n')
      self.output.write(code[1])
      self.output.write('\n')
      self.output.write(code[2])
      self.output.write(main_code_preamble_A)
      self.output.write(code[3])
      self.output.write(main_code_preamble_B)
      self.output.write(main_code_postamble)
