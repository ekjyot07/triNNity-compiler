preamble = '''#include <chrono>
#include <iostream>
#include <algorithm>
#include <triNNity/layer.h>
#include <triNNity/generic/layer.h>
#include <triNNity/dense/cpu/layer.h>

#ifndef GEMM_TYPE
#define GEMM_TYPE triNNity::GEMM_BLAS
#endif

#ifndef GEMV_TYPE
#define GEMV_TYPE triNNity::GEMV_BLAS
#endif

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

class TrinnityRuntime(object):

    def __init__(self, output):
        self.output = output

    def generate(self, code, transformer):
      self.output.write(preamble)
      self.output.write('\n')
      self.output.write(code[0])
      self.output.write('\n')
      self.output.write('int main(int argc, char **argv) {\n')
      self.output.write('\n')
      self.output.write(code[1])
      self.output.write('  data = new ACTIVATION_TYPE[{}]();'.format(transformer.data_size))
      self.output.write('\n  ')
      self.output.write('\n  '.join(map(str, (list(map(lambda x: '{} = new {}[{}]();'.format(x[0], x[2], x[3]), transformer.declarations))))))
      self.output.write('\n')
      self.output.write('\n')
      self.output.write(code[2])
      self.output.write(main_code_preamble_A)
      self.output.write(code[3])
      self.output.write(main_code_preamble_B)
      self.output.write('\n')
      self.output.write('  std::cerr << "Classification:" << std::endl;')
      self.output.write('  std::for_each(' + str(transformer.output_node_name) + '->output, ' + str(transformer.output_node_name) + '->output+' + str(transformer.labels) + ', [](auto &x){ std::cerr << x << std::endl; });')
      self.output.write('  delete [] data;')
      self.output.write(main_code_postamble)
