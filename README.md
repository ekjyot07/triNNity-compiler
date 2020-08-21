# triNNity-compiler

Originally forked from https://gitlab.scss.tcd.ie/software-tools-group/cnn-research/triNNity-compiler

Compile [Caffe](https://github.com/BVLC/caffe/) models to code.

Currently supported backends:

Python:

- [tensorflow](https://github.com/tensorflow/tensorflow)

C++:

- [triNNity](https://bitbucket.org/STG-TCD/triNNity)
- [mkldnn](https://github.com/intel/mkl-dnn)
- [armcl](https://github.com/ARM-software/ComputeLibrary)

## Usage

```
triNNity-compiler [-h] [--model MODEL] [--weights WEIGHTS]
                       [--data-output DATA_OUTPUT]
                       [--code-output CODE_OUTPUT]
                       [--topology-output TOPOLOGY_OUTPUT]
                       [--layers-output LAYERS_OUTPUT]
                       [--constraints-output CONSTRAINTS_OUTPUT]
                       [--backend BACKEND] [-p PHASE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Input model definition (.prototxt) path
  --weights WEIGHTS     Input model data (.caffemodel) path
  --data-output DATA_OUTPUT
                        Place converted weights in this directory
  --code-output CODE_OUTPUT
                        Generate source code into this file
  --topology-output TOPOLOGY_OUTPUT
                        Generate topology description into this file
  --layers-output LAYERS_OUTPUT
                        Generate layer description into this file
  --constraints-output CONSTRAINTS_OUTPUT
                        Generate layer constraints into this file
  --backend BACKEND     Which backend to use for code generation
  -p PHASE, --phase PHASE
                        The phase to convert: test (default) or train

```
