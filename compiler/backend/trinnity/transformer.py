import numpy as np

from ...util.errors import CompilerError, print_stderr
from ...frontend.graph import IRGraphBuilder, IRNodeMapper
from ...frontend.layers import LayerKind
from ...util.transformers import (DataInjector, DataReshaper, NodeRenamer, ReLUFuser, BatchNormScaleBiasFuser, BatchNormPreprocessor, ParameterNamer)

class TrinnityNode(object):
    '''An intermediate representation for Trinnity operations.'''

    def __init__(self, op, *args, **kwargs):
        # A string corresponding to the Trinnity operation
        self.op = op
        self.orig_op = op
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
        # Caffe has some layers that we need to process but basically ignore
        self.magic_layers = ['data', 'label', 'accuracy', 'softmax_loss']

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

        # Collect allocations
        allocs = []

        # Select the triNNity primitive corresponding to this op
        if (self.op == 'conv'):
            self.op = 'triNNity::generic::layer::GenericFusedConvolutionalLayer'
            act = 'triNNity::ACTIVATION_NONE'
            if has_relu:
              act = 'triNNity::ACTIVATION_RELU'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            # Set up weights buffer
            self.weights_buffer_name = self.node.name + '_weights'
            self.weights_buffer = 'WEIGHT_TYPE' + ' * ' + self.weights_buffer_name + ' = new ' + 'WEIGHT_TYPE' + '[' + str(int(args[0])*int(args[3])*int(args[3])*int(args[6])) + '];'
            allocs += [self.weights_buffer + '\n']

            # Set up bias buffer
            self.bias_buffer_name = self.node.name + '_bias'
            self.bias_buffer = 'WEIGHT_TYPE' + ' * ' + self.bias_buffer_name + ' = new ' + 'WEIGHT_TYPE' + '[' + str(int(args[6])) + '];'
            allocs += [self.bias_buffer + '\n']

            # Set up output buffer
            self.output_buffer_name = self.node.name + '_output'
            self.output_buffer = 'ACTIVATION_TYPE' + ' * ' + self.output_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[6])*int(args[1])*int(args[2])) + '];'
            allocs += [self.output_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE', 'LAYER_'+self.node.name.upper()+'_METHOD', 'triNNity::GEMM_BLAS'] + args + ['LAYER_'+(self.node.name.upper())+'_IN_FMT', 'triNNity::BOUND_IMPLICIT_PAD', act])

        elif (self.op == 'relu'):
            self.op = 'triNNity::layer::ActivationLayer'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'max_pool'):
            self.op = 'triNNity::layer::PoolingLayer'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'avg_pool'):
            self.op = 'triNNity::layer::PoolingLayer'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'fc'):
            self.op = 'triNNity::layer::FCLayer'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            # Set up weights buffer
            self.weights_buffer_name = self.node.name + '_weights'
            self.weights_buffer = 'WEIGHT_TYPE' + ' * ' + self.weights_buffer_name + ' = new ' + 'WEIGHT_TYPE' + '[' + str(int(args[0])*int(args[0])*int(args[0])) + '];'
            allocs += [self.weights_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'softmax'):
            self.op = 'triNNity::layer::SoftmaxLayer'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'lrn'):
            self.op = 'triNNity::layer::ChannelwiseLRNLayer'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'concat'):
            self.op = 'triNNity::layer::ChannelwiseConcatLayer'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'batch_normalization'):
            self.op = 'triNNity::layer::BatchNormalizationLayer'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'multiply'):
            self.op = 'triNNity::layer::EltwiseLayer'
            self.elt_op = 'triNNity::ELTWISE_MUL'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'add'):
            self.op = 'triNNity::layer::EltwiseLayer'
            self.elt_op = 'triNNity::ELTWISE_ADD'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        elif (self.op == 'max'):
            self.op = 'triNNity::layer::EltwiseLayer'
            self.elt_op = 'triNNity::ELTWISE_MAX'

            # Set up input buffer
            if (self.op not in self.magic_layers):
                papa = self.node.get_only_parent()
                self.input_buffer_name = papa.name + '.output'
                self.input_buffer = ''
            else:
                self.input_buffer_name = self.node.name + '_input'
                self.input_buffer = 'ACTIVATION_TYPE' + ' * ' + self.input_buffer_name + ' = new ' + 'ACTIVATION_TYPE' + '[' + str(int(args[0])*int(args[1])*int(args[2])) + '];'
                allocs += [self.input_buffer + '\n']

            args = ', '.join(['ACTIVATION_TYPE', 'WEIGHT_TYPE', 'ACTIVATION_TYPE'] + args)

        else:
            if (self.op not in self.magic_layers):
                print_stderr('triNNity backend does not implement layer \'' + self.op + '\'')
            args = ''

        dynamic_args = []
        if self.input_buffer_name:
            dynamic_args += [self.input_buffer_name]
        if self.weights_buffer_name:
            dynamic_args += [self.weights_buffer_name]
        if self.bias_buffer_name:
            dynamic_args += [self.bias_buffer_name]
        if self.output_buffer_name:
            dynamic_args += [self.output_buffer_name]

        outputs = []
        if (self.orig_op not in self.magic_layers):
          outputs += [self.op + '<' + args + '>' + ' ' + self.node.name + '(' + ', '.join(dynamic_args) + ');']

        return (allocs, outputs)


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
        return kernel_params

    def map_convolution(self, node):
        kernel_params = self.get_kernel_params(node)
        kwargs = {}
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
            raise CompilerError('Unsupported pooling type.')
        kernel_params = self.get_kernel_params(node)
        return TrinnityNode(pool_op, kernel_params.kernel_h, kernel_params.kernel_w,
                            kernel_params.stride_h, kernel_params.stride_w,
                            kernel_params.pad_h, kernel_params.pad_w)

    def map_inner_product(self, node):
        assert node.parameters.axis == 1
        assert node.parameters.bias_term == True
        return MaybeActivated(node)('fc', node.parameters.num_output)

    def map_softmax(self, node):
        return TrinnityNode('softmax')

    def map_lrn(self, node):
        params = node.parameters
        assert params.local_size % 2 == 1
        # Caffe scales by (alpha/(2*n+1))
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
        self.collected_allocations = []
        self.collected_code = []

    def indent(self):
        self.prefix += self.tab

    def outdent(self):
        self.prefix = self.prefix[:-len(self.tab)]

    def statement(self, s):
        return self.prefix + s + '\n'

    def emit_imports(self, name):
        return self.statement('#include "'+name+'.h"')

    def emit_parents(self, chain):
        assert len(chain)
        sep = '\n' + self.prefix
        s = sep.join(["'%s'" % parent.name for parent in chain[0].node.parents])
        return self.statement(s)

    def emit_node(self, node):
        (allocs, code) = node.emit()
        self.collected_allocations += allocs
        self.collected_code += list(map(lambda x: self.statement(str(x)), code))

    def emit(self, name, chains):
        s = self.emit_imports(name)

        for chain in chains:
            for node in chain:
                self.emit_node(node)

        s += ''.join(self.collected_allocations)
        s += '\n\n'
        s += ''.join(self.collected_code)
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

    def transform_source(self):
        if self.source is None:
            mapper = TrinnityMapper(self.graph)
            chains = mapper.map()
            emitter = TrinnityEmitter()
            self.source = emitter.emit(self.graph.name, chains)
        return [self.source]
