import numpy as np

from ...util.errors import CompilerError, print_stderr
from ...frontend.graph import IRGraphBuilder, IRNodeMapper
from ...frontend.layers import LayerKind
from ...util.transformers import (DataInjector, DataReshaper, NodeRenamer, ReLUFuser, BatchNormScaleBiasFuser, BatchNormPreprocessor, ParameterNamer)

def get_padding_type(kernel_params, input_shape, output_shape):
    '''Translates Caffe's numeric padding to one of ('SAME', 'VALID').
    Caffe supports arbitrary padding values, while Trinnity only
    supports 'SAME' and 'VALID' modes. So, not all Caffe paddings
    can be translated to Trinnity. There are some subtleties to
    how the padding edge-cases are handled. These are described here:
    https://github.com/Yangqing/caffe2/blob/master/caffe2/proto/caffe2_legacy.proto
    '''
    k_h, k_w, s_h, s_w, p_h, p_w = kernel_params
    s_o_h = np.ceil(input_shape.height / float(s_h))
    s_o_w = np.ceil(input_shape.width / float(s_w))
    if (output_shape.height == s_o_h) and (output_shape.width == s_o_w):
        return 'SAME'
    v_o_h = np.ceil((input_shape.height - k_h + 1.0) / float(s_h))
    v_o_w = np.ceil((input_shape.width - k_w + 1.0) / float(s_w))
    if (output_shape.height == v_o_h) and (output_shape.width == v_o_w):
        return 'VALID'
    return None

class TrinnityNode(object):
    '''An intermediate representation for Trinnity operations.'''

    def __init__(self, op, *args, **kwargs):
        # A string corresponding to the Trinnity operation
        self.op = op
        # Positional arguments for the operation
        self.args = args
        # Keyword arguments for the operation
        self.kwargs = kwargs
        # The source Caffe node
        self.node = None
        # The name/decl of the input buffer
        self.input_buffer = None
        self.input_buffer_name = None
        # The name/decl of the weights buffer
        self.weights_buffer = None
        self.weights_buffer_name = None
        # The name/decl of the bias buffer
        self.bias_buffer = None
        self.bias_buffer_name = None
        # The name/decl of the output buffer (only if not an in-place layer)
        self.output_buffer = None
        self.output_buffer_name = None

    def format(self, arg):
        '''Returns a string representation for the given value.'''
        return "%s" % str(arg)

    def pair(self, key, value):
        '''Returns key=formatted(value).'''
        return '%s=%s' % (key, self.format(value))

    def emit(self):
        '''Emits the Python source for this node.'''
        # Format positional arguments
        args = list(map(self.format, self.args))

        has_relu = 'relu' in self.kwargs and self.kwargs['relu']

        # Format any keyword arguments
        # if self.kwargs:
            # args += [(self.pair(k, v)) for k, v in self.kwargs.items() if k != 'relu']

        # Select the triNNity primitive corresponding to this op
        if (self.op == 'conv'):
            self.op = 'triNNity::generic::layer::GenericFusedConvolutionalLayer'
            act = 'triNNity::ACTIVATION_NONE'
            if has_relu:
              act = 'triNNity::ACTIVATION_RELU'
            self.input_buffer_name = self.node.name + '_input'
            self.weights_buffer_name = self.node.name + '_weights'
            self.bias_buffer_name = self.node.name + '_bias'
            self.output_buffer_name = self.node.name + '_output'

            self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
            self.weights_buffer = 'WEIGHT_TYPE' + ' * ' + self.weights_buffer_name + '_weights' + ' = new ' + 'WEIGHT_TYPE' + '[' + str(int(args[0])*int(args[3])*int(args[3])*int(args[6])) + '];'
            self.bias_buffer = 'WEIGHT_TYPE' + ' * ' + self.bias_buffer_name + '_biases' + ' = new ' + 'WEIGHT_TYPE' + '[' + str(int(args[6])) + '];'
            self.output_buffer = 'ACTIVATION_TYPE' + ' * ' + self.output_buffer_name + '_output' + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[6])*int(args[1])*int(args[2])) + '];'
            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE', 'triNNity::generic::CONV_DIRECT_SUM_2D', 'triNNity::GEMM_BLOCKED'] + args + ['triNNity::CHW', 'triNNity::BOUND_IMPLICIT_PAD', act, 'GEMM_BLOCK_I', 'GEMM_BLOCK_J', 'GEMM_BLOCK_K'])
        else:
            args = ', '.join(args)

        outputs = []
        if self.input_buffer:
            outputs += [self.input_buffer]
        if self.weights_buffer:
            outputs += [self.weights_buffer]
        if self.bias_buffer:
            outputs += [self.bias_buffer]
        if self.output_buffer:
            outputs += [self.output_buffer]

        dynamic_args = []
        if self.input_buffer_name:
            dynamic_args += [self.input_buffer_name]
        if self.weights_buffer_name:
            dynamic_args += [self.weights_buffer_name]
        if self.bias_buffer_name:
            dynamic_args += [self.bias_buffer_name]
        if self.output_buffer_name:
            dynamic_args += [self.output_buffer_name]

        outputs += [self.op + '<' + args + '>' + ' ' + self.node.name + '(' + ', '.join(dynamic_args) + ');']

        return outputs


class MaybeActivated(object):

    def __init__(self, node, default=True):
        self.inject_kwargs = {}
        if node.metadata.get('relu', False) != default:
            self.inject_kwargs['relu'] = not default
        else:
            self.inject_kwargs['relu'] = default

    def __call__(self, *args, **kwargs):
        kwargs.update(self.inject_kwargs)
        return TrinnityNode(*args, **kwargs)


class TrinnityMapper(IRNodeMapper):

    def get_kernel_params(self, node):
        kernel_params = node.layer.kernel_parameters
        input_shape = node.get_only_parent().output_shape
        padding = get_padding_type(kernel_params, input_shape, node.output_shape)
        # Only emit the padding if it's not the default value.
        padding = {'padding': padding} if padding != 'SAME' else {}
        return (kernel_params, padding)

    def map_convolution(self, node):
        (kernel_params, kwargs) = self.get_kernel_params(node)
        k_h = kernel_params.kernel_h
        k_w = kernel_params.kernel_w
        s_h = kernel_params.stride_h
        s_w = kernel_params.stride_w
        c_o = node.output_shape[1]
        c_i = node.parents[0].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        h_o = int(h_i / s_h)
        w_o = int(w_i / s_w)
        group = node.parameters.group
        if group != 1:
            kwargs['group'] = group
        if not node.parameters.bias_term:
            kwargs['biased'] = False

        return MaybeActivated(node)('conv', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o, **kwargs)

    def map_relu(self, node):
        return TrinnityNode('relu')

    def map_pooling(self, node):
        pool_type = node.parameters.pool
        if pool_type == 0:
            pool_op = 'max_pool'
        elif pool_type == 1:
            pool_op = 'avg_pool'
        else:
            # Stochastic pooling, for instance.
            raise CompilerError('Unsupported pooling type.')
        (kernel_params, padding) = self.get_kernel_params(node)
        return TrinnityNode(pool_op, kernel_params.kernel_h, kernel_params.kernel_w,
                              kernel_params.stride_h, kernel_params.stride_w, **padding)

    def map_inner_product(self, node):
        #TODO: Axis
        assert node.parameters.axis == 1
        #TODO: Unbiased
        assert node.parameters.bias_term == True
        return MaybeActivated(node)('fc', node.parameters.num_output)

    def map_softmax(self, node):
        return TrinnityNode('softmax')

    def map_softmax_with_loss(self, node):
        return TrinnityNode('softmax_loss')

    def map_accuracy(self, node):
        return TrinnityNode('accuracy')

    def map_lrn(self, node):
        params = node.parameters
        # The window size must be an odd value. For a window
        # size of (2*n+1), Trinnity defines depth_radius = n.
        assert params.local_size % 2 == 1
        # Caffe scales by (alpha/(2*n+1)), whereas Trinnity
        # just scales by alpha (as does Krizhevsky's paper).
        # We'll account for that here.
        alpha = params.alpha / float(params.local_size)
        return TrinnityNode('lrn', int(params.local_size / 2), alpha, params.beta)

    def map_concat(self, node):
        axis = (2, 3, 1, 0)[node.parameters.axis]
        return TrinnityNode('concat', axis)

    def map_dropout(self, node):
        return TrinnityNode('dropout', node.parameters.dropout_ratio)

    def map_batch_norm(self, node):
        scale_offset = len(node.data) == 4
        kwargs = {} if scale_offset else {'scale_offset': False}
        return MaybeActivated(node, default=False)('batch_normalization', **kwargs)

    def map_eltwise(self, node):
        operations = {0: 'multiply', 1: 'add', 2: 'max'}
        op_code = node.parameters.operation
        try:
            return TrinnityNode(operations[op_code])
        except KeyError:
            raise CompilerError('Unknown elementwise operation: {}'.format(op_code))

    def commit(self, chains):
        return chains


class TrinnityEmitter(object):

    def __init__(self, tab=None):
        self.tab = tab or ' ' * 2
        self.prefix = ''

    def indent(self):
        self.prefix += self.tab

    def outdent(self):
        self.prefix = self.prefix[:-len(self.tab)]

    def statement(self, s):
        return self.prefix + s + '\n'

    def emit_imports(self):
        return self.statement('#include <triNNity/generic/layer.h>\n')

    def emit_parents(self, chain):
        assert len(chain)
        sep = '\n' + self.prefix
        s = sep.join(["'%s'" % parent.name for parent in chain[0].node.parents])
        return self.statement(s)

    def emit_node(self, node):
        return ''.join(list(map(lambda x: self.statement(str(x)), node.emit())))

    def emit(self, name, chains):
        s = self.emit_imports()
        blocks = []
        for chain in chains:
            b = ''
            b += self.emit_parents(chain)
            for node in chain:
                b += self.emit_node(node)
            blocks.append(b[:-1] + ')')
        s = s + '\n\n'.join(blocks)
        return s


class TrinnityTransformer(object):

    def __init__(self, def_path, data_path, verbose=True, phase='test'):
        self.verbose = verbose
        self.phase = phase
        self.load(def_path, data_path, phase)
        self.params = None
        self.source = None

    def load(self, def_path, data_path, phase):
        # Build the graph
        graph = IRGraphBuilder(def_path, phase).build()

        if data_path is not None:
            # Load and associate learned parameters
            graph = DataInjector(def_path, data_path)(graph)

        # Transform the graph
        transformers = [
            # Fuse split batch normalization layers
            BatchNormScaleBiasFuser(),

            # Fuse ReLUs
            # TODO: Move non-linearity application to layer wrapper, allowing
            # any arbitrary operation to be optionally activated.
            ReLUFuser(allowed_parent_types=[LayerKind.Convolution, LayerKind.InnerProduct,
                                            LayerKind.BatchNorm]),

            # Rename nodes
            # Slashes are used for scoping in Trinnity. Replace slashes
            # in node names with underscores.
            # (Caffe's GoogLeNet implementation uses slashes)
            NodeRenamer(lambda node: node.name.replace('/', '_'))
        ]
        self.graph = graph.transformed(transformers)

        # Display the graph
        if self.verbose:
            print_stderr(self.graph)

    def transform_data(self):
        if self.params is None:
            transformers = [

                # Reshape the parameters to Trinnity's ordering
                DataReshaper({
                    # (c_o, c_i, h, w) -> (h, w, c_i, c_o)
                    LayerKind.Convolution: (2, 3, 1, 0),

                    # (c_o, c_i) -> (c_i, c_o)
                    LayerKind.InnerProduct: (1, 0)
                }),

                # Pre-process batch normalization data
                BatchNormPreprocessor(),

                # Convert parameters to dictionaries
                ParameterNamer(),
            ]
            self.graph = self.graph.transformed(transformers)
            self.params = {node.name: node.data for node in self.graph.nodes if node.data}
        return self.params

    def transform_source(self):
        if self.source is None:
            mapper = TrinnityMapper(self.graph)
            chains = mapper.map()
            emitter = TrinnityEmitter()
            self.source = emitter.emit(self.graph.name, chains)
        return self.source
