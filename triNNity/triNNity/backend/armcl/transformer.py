import math
import numpy as np

from triNNity.util.errors import CompilerError, print_stderr
from triNNity.frontend.graph import IRGraphBuilder, IRNodeMapper
from triNNity.frontend.layers import LayerKind
from triNNity.util.transformers import (DataInjector, DataReshaper, NodeRenamer, ReLUFuser, BatchNormScaleBiasFuser, BatchNormPreprocessor, ParameterNamer, ConcatTreeSplitter)

class ARMCLNode(object):

    def __init__(self, op, *args, **kwargs):

        self.op = op
        self.orig_op = op

        self.args = args
        self.kwargs = kwargs

        self.node = None

        self.output_buffer = None
        self.output_buffer_name = None

        self.magic_layers = ['data', 'label']

    def format(self, arg):
        return "%s" % str(arg)

    def pair(self, key, value):
        return '%s=%s' % (key, self.format(value))

    def emit(self, graphName):

        args = list(map(self.format, self.args))  # formats all the arguments
        has_relu = 'relu' in self.kwargs and self.kwargs['relu']

        has_group = 'group' in self.kwargs and self.kwargs['group']

        # Collect allocations
        decls = []



        if(self.op == 'conv'):
            self.op = 'ConvolutionLayer'

            # args = ', '.join([str(int(args[3])), str(int(args[9]))+'tatti'])
            args = ', '.join([str(int(args[3]))+'U', str(int(args[3]))+'U', str(int(args[6]))+'U', 'get_weights_accessor(data_path, "/cnn_data/' + graphName.lower() + '_model/' + self.node.name.lower() + '_w.npy", weights_layout)'])
            #  + [
            #                  'get_weights_accessor(data_path, "/cnn_data/'+ graphName.lower() +'_model/'+(self.node.name())+'_b.npy"), PadStrideInfo(' + str(int(args[5])), str(int(args[6])), str(int(args[9])), str(int(args[9])) + ')'])
            # if (self.kwargs['group'] != 1):
            #     args.append(',' + self.kwargs['group'] + ')') 

        elif (self.op == 'relu'):
            self.op = 'ActivationLayer'

            args = 'ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)'

        elif (self.op == 'max_pool'):
            self.op = 'PoolingLayer'

            args = ', '.join(['PoolingLayerInfo(PoolingType::' + 'MAX', str(int(args[3])),
                             'PadStrideInfo(' + str(int(args[4])), str(int(args[5])), str(int(args[9])), str(int(args[9])) + ')'])

        elif (self.op == 'avg_pool'):
            self.op = 'PoolingLayer'

            # CHANGE THIS
            args = ', '.join(['PoolingLayerInfo(PoolingType::' + 'MAX', str(int(args[3])),
                             'PadStrideInfo(' + str(int(args[4])), str(int(args[5])), str(int(args[9])), str(int(args[9])) + '),'])

        elif (self.op == 'fc'):
            self.op = 'FullyConnectedLayer'

            args = ', '.join([str(int(args[3])) + 'U', 'get_weights_accessor(data_path, "/cnn_data/' + graphName.lower() + '_model/' + self.node.name.lower() + '_w.npy", weights_layout)'] + [
                             'get_weights_accessor(data_path, "/cnn_data/'+ graphName.lower() +'_model/' + self.node.name.lower() + '_b.npy")'])

        elif (self.op == 'softmax'):
            self.op = 'SoftmaxLayer'
            args = ''


        elif (self.op == 'lrn'):
            self.op = 'NormalizationLayer'
            args = ', '.join(['NormalizationLayerInfo(NormType::' + 'CROSS_MAP', str(self.kwargs['size']), str(self.kwargs['alpha']), str(self.kwargs['beta']) + 'f))' ])

            

        elif (self.op == 'concat'):
            self.op='triNNity::layer::ChannelwiseConcatLayer'
            args = ''

        elif (self.op == 'batch_normalization'):
            self.op='triNNity::layer::BatchNormalizationLayer'
            args = ''


        elif (self.op == 'multiply'):
            self.op='triNNity::layer::EltwiseLayer'
            self.elt_op='triNNity::ELTWISE_MUL'
            args = ''


        elif (self.op == 'add'):
            self.op='triNNity::layer::EltwiseLayer'
            self.elt_op='triNNity::ELTWISE_ADD'
            args = ''


        elif (self.op == 'max'):
            self.op='triNNity::layer::EltwiseLayer'
            self.elt_op='triNNity::ELTWISE_MAX'
            args = ''

        else:
            if (self.op not in self.magic_layers):
                print_stderr(
                    'triNNity backend does not implement layer \'' + self.op + '\'')
            args=''

        outputs = []
        print('[INFO]')
        print(args)
        if (self.orig_op not in self.magic_layers):
            outputs += ['<<' + self.op + '(' + args + ')' + '.set_name(' + self.node.name.lower() + ')']

        return(decls, outputs)


class MaybeActivated(object):

    def __init__(self, node, default=True):
        self.inject_kwargs = {}
        if node.metadata.get('relu', False) != default:
            self.inject_kwargs['relu'] = not default
        else:
            self.inject_kwargs['relu'] = default

    def __call__(self, *args, **kwargs):
        kwargs.update(self.inject_kwargs)
        return ARMCLNode(*args, **kwargs)


class ARMCLMapper(IRNodeMapper):

    def get_kernel_params(self, node):
        kernel_params = node.layer.kernel_parameters
        return kernel_params

    def map_convolution(self, node):
        kernel_params = self.get_kernel_params(node)
        kwargs = {}
        # k_h = kernel_params.kernel_h
        k_w = kernel_params.kernel_w
        s_h = kernel_params.stride_h
        s_w = kernel_params.stride_w
        p_w = kernel_params.pad_w
        #p_h = kernel_params.pad_h
        c_o = node.output_shape[1]
        c_i = node.parents[0].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        h_o = int(math.ceil(h_i / s_h))
        w_o = int(math.ceil(w_i / s_w))
        group = node.parameters.group
        if group != 1:
            kwargs['group'] = group
        if not node.parameters.bias_term:
            kwargs['biased'] = False

        return MaybeActivated(node)('conv', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o, p_w, **kwargs)

    def map_relu(self, node):
        c_i = node.parents[0].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        return ARMCLNode('relu', c_i, w_i, h_i, 'triNNity::ACTIVATION_RELU')

    def map_pooling(self, node):
        pool_type = node.parameters.pool
        kernel_params = self.get_kernel_params(node)
        # k_h = kernel_params.kernel_h
        k_w = kernel_params.kernel_w
        s_h = kernel_params.stride_h
        s_w = kernel_params.stride_w
        c_o = node.output_shape[1]
        c_i = node.parents[0].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        h_o = int(math.ceil(h_i / s_h))
        w_o = int(math.ceil(w_i / s_w))
        p_w = kernel_params.pad_w
        #p_h = kernel_params.pad_h

        if pool_type == 0:
            pool_op = 'max_pool'
        elif pool_type == 1:
            pool_op = 'avg_pool'
        else:
            raise CompilerError('Unsupported pooling type.')
        kernel_params = self.get_kernel_params(node)
        return ARMCLNode(pool_op, c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o, p_w)

    def map_inner_product(self, node):
        assert node.parameters.axis == 1
        assert node.parameters.bias_term == True
        c_i = node.parents[0].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        return MaybeActivated(node)('fc', c_i, w_i, h_i, node.parameters.num_output)

    def map_softmax(self, node):
        return ARMCLNode('softmax', node.parents[0].output_shape[1])

    def map_lrn(self, node):
        params = node.parameters
        assert params.local_size % 2 == 1
        c_i = node.parents[0].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        kwargs = {}
        kwargs['alpha'] = params.alpha
        kwargs['beta'] = params.beta
        kwargs['size'] = params.local_size
        return ARMCLNode('lrn', c_i, w_i, h_i, 'triNNity::layout::CHW', **kwargs)

    def map_concat(self, node):
        axis = [node.parameters.axis]
        c_i_0 = node.parents[0].output_shape[1]
        c_i_1 = node.parents[1].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        if axis != [1]:
            raise CompilerError('Found concat node with unsupported join axis: %s' % axis)
        return ARMCLNode('concat', c_i_0, c_i_1, w_i, h_i, 'triNNity::layout::CHW')

    def map_dropout(self, node):
        return ARMCLNode('dropout', node.parameters.dropout_ratio)

    def map_batch_norm(self, node):
        c_i = node.parents[0].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        return MaybeActivated(node, default=False)('batch_normalization', c_i, w_i, h_i, 'triNNity::layout::CHW')

    def map_eltwise(self, node):
        operations = {0: 'multiply', 1: 'add', 2: 'max'}
        op_code = node.parameters.operation
        elt_count = 1
        for x in node.output_shape:
            elt_count *= x
        try:
            return ARMCLNode(operations[op_code], elt_count)
        except KeyError:
            raise CompilerError('Unknown elementwise operation: {}'.format(op_code))

    def commit(self, chains):
        return chains


class ARMCLEmitter(object):

    def __init__(self, tab=None):
        self.tab = tab or ' ' * 2
        self.prefix = ''
        self.collected_declarations = []
        self.collected_code = []
        self.collected_layers = []

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

    def emit_node(self, node, name):
        (decls, code) = node.emit(name)
        self.collected_declarations += decls
        self.collected_code += list(map(lambda x: self.statement(str(x)), code))
        self.collected_layers += [node.node.name + ".execute();"]

    def emit(self, name, chains):
        self.indent()

        for chain in chains:
            for node in chain:
                self.emit_node(node, name)

        c = self.prefix.join(self.collected_code)
        c += '\n'
        return [c]


class ARMCLTransformer(object):

    def __init__(self, def_path, data_path, verbose=True, phase='test'):
        self.verbose = verbose
        self.phase = phase
        self.load(def_path, data_path, phase)
        self.params = None
        self.sources = None

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
            # (Caffe's GoogLeNet implementation uses slashes)
            NodeRenamer(lambda node: node.name.replace('/', '_')),

            # Split concat operations into balanced binary trees
            ConcatTreeSplitter()
        ]
        self.graph = graph.transformed(transformers)

        topsorted_graph = self.graph.topologically_sorted()

        self.data_size = 1
        for x in topsorted_graph[0].output_shape:
            self.data_size *= x

        self.labels = 1
        for x in topsorted_graph[-1].output_shape:
            self.labels *= x

        self.output_node_name = topsorted_graph[-1].name

        # Display the graph
        if self.verbose:
            print_stderr(self.graph)

    def transform_source(self):
        if self.sources is None:
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
                # (Caffe's GoogLeNet implementation uses slashes)
                NodeRenamer(lambda node: node.name.replace('/', '_')),
            ]
            self.graph = self.graph.transformed(transformers)

            mapper = ARMCLMapper()
            chains = mapper.map(self.graph)
            emitter = ARMCLEmitter()
            self.sources = emitter.emit(self.graph.name, chains)
            self.declarations = emitter.collected_declarations
        return self.sources
