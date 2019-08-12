#include <chrono>
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


#include "AlexNet.h"


int main(int argc, char **argv) {

  ACTIVATION_TYPE * data;
  WEIGHT_TYPE * conv1_weights;
  WEIGHT_TYPE * conv1_bias;
  ACTIVATION_TYPE * conv1_output;
  WEIGHT_TYPE * conv2_weights;
  WEIGHT_TYPE * conv2_bias;
  ACTIVATION_TYPE * conv2_output;
  WEIGHT_TYPE * conv3_weights;
  WEIGHT_TYPE * conv3_bias;
  ACTIVATION_TYPE * conv3_output;
  WEIGHT_TYPE * conv4_weights;
  WEIGHT_TYPE * conv4_bias;
  ACTIVATION_TYPE * conv4_output;
  WEIGHT_TYPE * conv5_weights;
  WEIGHT_TYPE * conv5_bias;
  ACTIVATION_TYPE * conv5_output;
  WEIGHT_TYPE * fc6_weights;
  WEIGHT_TYPE * fc7_weights;
  WEIGHT_TYPE * fc8_weights;

  data = new ACTIVATION_TYPE[150528]();
  conv1_weights = new WEIGHT_TYPE[34848]();
  conv1_bias = new WEIGHT_TYPE[96]();
  conv1_output = new ACTIVATION_TYPE[301056]();
  conv2_weights = new WEIGHT_TYPE[614400]();
  conv2_bias = new WEIGHT_TYPE[256]();
  conv2_output = new ACTIVATION_TYPE[186624]();
  conv3_weights = new WEIGHT_TYPE[884736]();
  conv3_bias = new WEIGHT_TYPE[384]();
  conv3_output = new ACTIVATION_TYPE[64896]();
  conv4_weights = new WEIGHT_TYPE[1327104]();
  conv4_bias = new WEIGHT_TYPE[384]();
  conv4_output = new ACTIVATION_TYPE[64896]();
  conv5_weights = new WEIGHT_TYPE[884736]();
  conv5_bias = new WEIGHT_TYPE[256]();
  conv5_output = new ACTIVATION_TYPE[43264]();
  fc6_weights = new WEIGHT_TYPE[37748736]();
  fc7_weights = new WEIGHT_TYPE[16777216]();
  fc8_weights = new WEIGHT_TYPE[4096000]();

  triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV1_METHOD, GEMM_TYPE, 3, 224, 224, 11, 11, 4, 4, 96, 56, 56, LAYER_CONV1_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU> *conv1 = new triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV1_METHOD, GEMM_TYPE, 3, 224, 224, 11, 11, 4, 4, 96, 56, 56, LAYER_CONV1_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU>(data, conv1_weights, conv1_bias, conv1_output);
  triNNity::layer::ChannelwiseLRNLayer<ACTIVATION_TYPE, 96, 54, 54, triNNity::layout::CHW> *norm1 = new triNNity::layer::ChannelwiseLRNLayer<ACTIVATION_TYPE, 96, 54, 54, triNNity::layout::CHW>(conv1->output, 0, 5, 9.999999747378752e-05, 0.75);
  triNNity::layer::PoolingLayer<ACTIVATION_TYPE, triNNity::WINDOW_MAXPOOL, 96, 54, 54, 3, 3, 2, 2, 96, 27, 27> *pool1 = new triNNity::layer::PoolingLayer<ACTIVATION_TYPE, triNNity::WINDOW_MAXPOOL, 96, 54, 54, 3, 3, 2, 2, 96, 27, 27>(norm1->output);
  triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV2_METHOD, GEMM_TYPE, 96, 27, 27, 5, 5, 1, 1, 256, 27, 27, LAYER_CONV2_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU> *conv2 = new triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV2_METHOD, GEMM_TYPE, 96, 27, 27, 5, 5, 1, 1, 256, 27, 27, LAYER_CONV2_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU>(pool1->output, conv2_weights, conv2_bias, conv2_output);
  triNNity::layer::ChannelwiseLRNLayer<ACTIVATION_TYPE, 256, 27, 27, triNNity::layout::CHW> *norm2 = new triNNity::layer::ChannelwiseLRNLayer<ACTIVATION_TYPE, 256, 27, 27, triNNity::layout::CHW>(conv2->output, 0, 5, 9.999999747378752e-05, 0.75);
  triNNity::layer::PoolingLayer<ACTIVATION_TYPE, triNNity::WINDOW_MAXPOOL, 256, 27, 27, 3, 3, 2, 2, 256, 14, 14> *pool2 = new triNNity::layer::PoolingLayer<ACTIVATION_TYPE, triNNity::WINDOW_MAXPOOL, 256, 27, 27, 3, 3, 2, 2, 256, 14, 14>(norm2->output);
  triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV3_METHOD, GEMM_TYPE, 256, 13, 13, 3, 3, 1, 1, 384, 13, 13, LAYER_CONV3_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU> *conv3 = new triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV3_METHOD, GEMM_TYPE, 256, 13, 13, 3, 3, 1, 1, 384, 13, 13, LAYER_CONV3_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU>(pool2->output, conv3_weights, conv3_bias, conv3_output);
  triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV4_METHOD, GEMM_TYPE, 384, 13, 13, 3, 3, 1, 1, 384, 13, 13, LAYER_CONV4_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU> *conv4 = new triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV4_METHOD, GEMM_TYPE, 384, 13, 13, 3, 3, 1, 1, 384, 13, 13, LAYER_CONV4_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU>(conv3->output, conv4_weights, conv4_bias, conv4_output);
  triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV5_METHOD, GEMM_TYPE, 384, 13, 13, 3, 3, 1, 1, 256, 13, 13, LAYER_CONV5_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU> *conv5 = new triNNity::generic::layer::GenericFusedConvolutionalLayer<ACTIVATION_TYPE, WEIGHT_TYPE, ACTIVATION_TYPE, LAYER_CONV5_METHOD, GEMM_TYPE, 384, 13, 13, 3, 3, 1, 1, 256, 13, 13, LAYER_CONV5_IN_FMT, triNNity::BOUND_IMPLICIT_PAD, triNNity::ACTIVATION_RELU>(conv4->output, conv5_weights, conv5_bias, conv5_output);
  triNNity::layer::PoolingLayer<ACTIVATION_TYPE, triNNity::WINDOW_MAXPOOL, 256, 13, 13, 3, 3, 2, 2, 256, 7, 7> *pool5 = new triNNity::layer::PoolingLayer<ACTIVATION_TYPE, triNNity::WINDOW_MAXPOOL, 256, 13, 13, 3, 3, 2, 2, 256, 7, 7>(conv5->output);
  triNNity::layer::FCLayer<ACTIVATION_TYPE, WEIGHT_TYPE, GEMV_TYPE, 256, 6, 6, 4096> *fc6 = new triNNity::layer::FCLayer<ACTIVATION_TYPE, WEIGHT_TYPE, GEMV_TYPE, 256, 6, 6, 4096>(pool5->output, fc6_weights);
  triNNity::layer::FCLayer<ACTIVATION_TYPE, WEIGHT_TYPE, GEMV_TYPE, 4096, 1, 1, 4096> *fc7 = new triNNity::layer::FCLayer<ACTIVATION_TYPE, WEIGHT_TYPE, GEMV_TYPE, 4096, 1, 1, 4096>(fc6->output, fc7_weights);
  triNNity::layer::FCLayer<ACTIVATION_TYPE, WEIGHT_TYPE, GEMV_TYPE, 4096, 1, 1, 1000> *fc8 = new triNNity::layer::FCLayer<ACTIVATION_TYPE, WEIGHT_TYPE, GEMV_TYPE, 4096, 1, 1, 1000>(fc7->output, fc8_weights);
  triNNity::layer::SoftmaxLayer<ACTIVATION_TYPE, 1000> *prob = new triNNity::layer::SoftmaxLayer<ACTIVATION_TYPE, 1000>(fc8->output);

  unsigned times[NO_OF_RUNS];

  auto t1 = std::chrono::high_resolution_clock::now();

  for (unsigned i = 0; i < NO_OF_RUNS; i++) {
    conv1->execute();
    norm1->execute();
    pool1->execute();
    conv2->execute();
    norm2->execute();
    pool2->execute();
    conv3->execute();
    conv4->execute();
    conv5->execute();
    pool5->execute();
    fc6->execute();
    fc7->execute();
    fc8->execute();
    prob->execute();

    auto t2 = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    t1 = t2;
  }

  for (unsigned i = 0; i < NO_OF_RUNS; i++) {
    std::cout << times[i] << std::endl;
  }
  
  std::cerr << "Classification:" << std::endl;  std::for_each(prob->output, prob->output+1000, [](auto &x){ std::cerr << x << std::endl; });  delete [] data;
  return 0;
}
  