import numpy as np

from ...util.errors import CompilerError, print_stderr
from ...frontend.graph import IRGraphBuilder, IRNodeMapper
from ...frontend.layers import LayerKind
from ...util.transformers import (DataInjector, DataReshaper, NodeRenamer, ReLUFuser, BatchNormScaleBiasFuser, BatchNormPreprocessor, ParameterNamer)

class InfoNode(object):
    '''An intermediate representation for Info operations.'''

    def __init__(self, op, *args, **kwargs):
        # A string corresponding to the Info operation
        self.op = op
        self.orig_op = op
        # Positional arguments for the operation
        self.args = args
        # Keyword arguments for the operation
        self.kwargs = kwargs
        # The source Caffe node
        self.node = None
        # Caffe has some layers that we need to process but basically ignore
        self.magic_layers = ['data', 'label', 'accuracy', 'softmax_loss']

    def format(self, arg):
        '''Returns a string representation for the given value.'''
        return "%s" % str(arg)

    def pair(self, key, value):
        '''Returns key=formatted(value).'''
        return '%s=%s' % (key, self.format(value))

    def emit(self):
        '''Emits the topology source line for this node.'''

        has_relu = 'relu' in self.kwargs and self.kwargs['relu']

        # Collect out edges
        edges = []

        c_i = str(int(args[0]))
        h_i = str(int(args[1]))
        w_i = str(int(args[2]))
        k =   str(int(args[3]))
        s_w = str(int(args[4]))
        s_h = str(int(args[5]))
        c_o = str(int(args[6]))
        w_o = str(int(args[7]))
        h_o = str(int(args[8]))
        sparsity = '0'

        outputs = []
        if (self.orig_op not in self.magic_layers):
          outputs += [' '.join([c_o, c_i, s_w, w_i, h_i, k, sparsity])]

        return (edges, outputs)


class MaybeActivated(object):

    def __init__(self, node, default=True):
        self.inject_kwargs = {}
        if node.metadata.get('relu', False) != default:
            self.inject_kwargs['relu'] = not default
        else:
            self.inject_kwargs['relu'] = default

    def __call__(self, *args, **kwargs):
        kwargs.update(self.inject_kwargs)
        return InfoNode(*args, **kwargs)


class InfoMapper(IRNodeMapper):

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
        return InfoNode('relu', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_pooling(self, node):
        return InfoNode('pooling', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_inner_product(self, node):
        return MaybeActivated(node)('fc', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_softmax(self, node):
        return InfoNode('softmax', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_softmax_with_loss(self, node):
        return InfoNode('softmax_loss', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_accuracy(self, node):
        return InfoNode('accuracy', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_lrn(self, node):
        return InfoNode('lrn', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_concat(self, node):
        return InfoNode('concat', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_dropout(self, node):
        return InfoNode('dropout', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_batch_norm(self, node):
        return MaybeActivated(node)('batch_normalization', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)

    def map_eltwise(self, node):
        operations = {0: 'multiply', 1: 'add', 2: 'max'}
        op_code = node.parameters.operation
        try:
            return InfoNode(operations[op_code], c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o)
        except KeyError:
            raise CompilerError('Unknown elementwise operation: {}'.format(op_code))

    def commit(self, chains):
        return chains


class InfoEmitter(object):

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

    def emit_imports(self):
        return self.statement('NODES PARAMS EDGES')

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
        s = self.emit_imports()

        for chain in chains:
            for node in chain:
                self.emit_node(node)

        s += ''.join(self.collected_allocations)
        s += '\n\n'
        s += ''.join(self.collected_code)
        return s


class InfoTransformer(object):

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
            # Slashes are used for scoping in Info. Replace slashes
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

                # Reshape the parameters to Info's ordering
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
            mapper = InfoMapper(self.graph)
            chains = mapper.map()
            emitter = InfoEmitter()
            self.toposource = emitter.emit(self.graph.name, chains)
            self.layersource = '\n'.join({node.name for node in self.graph.nodes})
        return [self.toposource, self.layersource]
