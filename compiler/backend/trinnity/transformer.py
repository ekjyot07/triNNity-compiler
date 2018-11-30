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
        self.args = args
        self.kwargs = kwargs
        # The source Caffe node
        self.node = None
        # Whether this operation has weights
        self.has_weights = False

    def emit(self):
        '''Emits the C++ code for this node.'''
        # Format static arguments
        template_args = list(map(str, self.args))
        template_args = ', '.join(template_args)
        # Format dynamic arguments
        dynamic_args = [self.kwargs['input']]

        if (self.has_weights):
            dynamic_args += [self.kwargs['weights']]

        if (self.kwargs['biased']):
            dynamic_args += [self.kwargs['biases']]

        dynamic_args = ', '.join(dynamic_args)

        return '%s<%s> %s(%s);' % (self.op, template_args, self.node.name, dynamic_args)


class MaybeActivated(object):

    def __init__(self, node, default=True):
        self.inject_kwargs = {}
        if node.metadata.get('relu', False) != default:
            self.inject_kwargs['relu'] = not default

    def __call__(self, *args, **kwargs):
        kwargs.update(self.inject_kwargs)
        return TrinnityNode(*args, **kwargs)


class TrinnityMapper(IRNodeMapper):

    def map_convolution(self, node):
        kernel_params = node.layer.kernel_parameters
        k = 3
        h = kernel_params.kernel_h
        w = kernel_params.kernel_w
        m = node.output_shape[1]
        c = node.parents[0].output_shape[1]
        sw = kernel_params.stride_w
        sh = kernel_params.stride_h
        ow = w / sw
        oh = h / sh

        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        group = node.parameters.group
        if group != 1:
            kwargs['group'] = group

        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'

        return MaybeActivated(node)('triNNity::generic::layer::GenericFusedConvolutionalLayer', c, w, h, k, sw, sh, m, ow, oh, **kwargs)

    def map_relu(self, node):
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        kwargs['biased'] = False
        return TrinnityNode('relu', **kwargs)

    def map_pooling(self, node):
        pool_type = node.parameters.pool
        if pool_type == 0:
            pool_op = 'max_pool'
        elif pool_type == 1:
            pool_op = 'avg_pool'
        else:
            # Stochastic pooling, for instance.
            raise CompilerError('Unsupported pooling type.')
        kernel_params = node.layer.kernel_parameters
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        kwargs['biased'] = False
        return TrinnityNode(pool_op, kernel_params.kernel_h, kernel_params.kernel_w,
                              kernel_params.stride_h, kernel_params.stride_w, **kwargs)

    def map_inner_product(self, node):
        #TODO: Axis
        assert node.parameters.axis == 1
        #TODO: Unbiased
        assert node.parameters.bias_term == True
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        return MaybeActivated(node)('fc', node.parameters.num_output, **kwargs)

    def map_softmax(self, node):
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        return TrinnityNode('softmax', **kwargs)

    def map_softmax_with_loss(self, node):
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        return TrinnityNode('softmax_loss', **kwargs)

    def map_accuracy(self, node):
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        return TrinnityNode('accuracy', **kwargs)


    def map_lrn(self, node):
        params = node.parameters
        # The window size must be an odd value. For a window
        # size of (2*n+1), Trinnity defines depth_radius = n.
        assert params.local_size % 2 == 1
        # Caffe scales by (alpha/(2*n+1)), whereas Trinnity
        # just scales by alpha (as does Krizhevsky's paper).
        # We'll account for that here.
        alpha = params.alpha / float(params.local_size)
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        return TrinnityNode('lrn', int(params.local_size / 2), alpha, params.beta, **kwargs)

    def map_concat(self, node):
        axis = (2, 3, 1, 0)[node.parameters.axis]
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        return TrinnityNode('concat', axis, **kwargs)

    def map_dropout(self, node):
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        return TrinnityNode('dropout', node.parameters.dropout_ratio, **kwargs)

    def map_batch_norm(self, node):
        scale_offset = len(node.data) == 4
        kwargs = {} if scale_offset else {'scale_offset': False}
        kwargs += {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        return MaybeActivated(node, default=False)('batch_normalization', **kwargs)

    def map_eltwise(self, node):
        operations = {0: 'multiply', 1: 'add', 2: 'max'}
        op_code = node.parameters.operation
        kwargs = {'input':'input_arr', 'output':'output_arr', 'weights':'weights_arr'}
        if not node.parameters.bias_term:
            kwargs['biased'] = False
        else:
            kwargs['biased'] = True
            kwargs['biases'] = 'bias_arr'
        try:
            return TrinnityNode(operations[op_code], **kwargs)
        except KeyError:
            raise CompilerError('Unknown elementwise operation: {}'.format(op_code))

    def commit(self, chains):
        return chains


class TrinnityEmitter(object):

    def __init__(self, tab=None):
        self.tab = tab or ' ' * 4
        self.prefix = ''

    def indent(self):
        self.prefix += self.tab

    def outdent(self):
        self.prefix = self.prefix[:-len(self.tab)]

    def statement(self, s):
        return self.prefix + s + '\n'

    def emit_imports(self):
        return (self.statement('#include <triNNity/layer.h>') +
                self.statement('#include <triNNity/generic/layer.h>'))

    def emit_class_def(self, name):
        return self.statement('class %s(Network):' % (name))

    def emit_setup_def(self):
        return self.statement('def setup(self):')

    def emit_parents(self, chain):
        assert len(chain)
        s = '(self.feed('
        sep = ', \n' + self.prefix + (' ' * len(s))
        s += sep.join(["'%s'" % parent.name for parent in chain[0].node.parents])
        return self.statement(s + ')')

    def emit_node(self, node):
        return self.statement(' ' * 5 + '.' + node.emit())

    def emit(self, name, chains):
        s = self.emit_imports()
        s += self.emit_class_def(name)
        self.indent()
        s += self.emit_setup_def()
        self.indent()
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
            ReLUFuser(allowed_parent_types=[LayerKind.Convolution]),

            # Rename nodes
            # Replace slashes in node names with underscores.
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
