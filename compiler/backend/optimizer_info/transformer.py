import math
import numpy as np

from ...util.errors import CompilerError, print_stderr
from ...frontend.graph import IRGraphBuilder, IRNodeMapper
from ...frontend.layers import LayerKind
from ...util.transformers import (DataInjector, DataReshaper, NodeRenamer, ReLUFuser, BatchNormScaleBiasFuser, BatchNormPreprocessor, ParameterNamer, ConcatTreeSplitter)

magic_layers = ['data', 'label', 'accuracy', 'softmax_loss', 'loss', 'top-1', 'top-5']

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
        # The default constraints
        self.constraints = ('chw', '*', 'chw')

        if (op == 'conv'):
          self.constraints = ('*', '*', '*')

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

        c_i = str(int(self.args[0]))
        h_i = str(int(self.args[1]))
        w_i = str(int(self.args[2]))
        k =   str(int(self.args[3]))
        s_w = str(int(self.args[4]))
        s_h = str(int(self.args[5]))
        c_o = str(int(self.args[6]))
        w_o = str(int(self.args[7]))
        h_o = str(int(self.args[8]))
        sparsity = '0'

        outputs = []
        if (self.orig_op not in magic_layers):
          outputs += [' '.join([c_o, c_i, s_w, w_i, h_i, k, sparsity])]

        return (edges, outputs)

    def emit_constraints(self):
        '''Emits the constraints file line for this node.'''
        return ', '.join([self.node.name] + list(self.constraints))


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

    def map_convolution(self, node):
        kernel_params = node.layer.kernel_parameters
        k_h = kernel_params.kernel_h
        k_w = kernel_params.kernel_w
        s_h = kernel_params.stride_h
        s_w = kernel_params.stride_w
        c_i = node.parents[0].output_shape[1]
        h_i = node.parents[0].output_shape[2]
        w_i = node.parents[0].output_shape[3]
        c_o = node.output_shape[1]
        h_o = int(math.ceil(h_i / s_h))
        w_o = int(math.ceil(w_i / s_w))
        group = node.parameters.group

        kwargs = {}
        if group != 1:
            kwargs['group'] = group
        if not node.parameters.bias_term:
            kwargs['biased'] = False

        return MaybeActivated(node)('conv', c_i, w_i, h_i, k_w, s_w, s_h, c_o, w_o, h_o, **kwargs)

    def map_relu(self, node):
        return InfoNode('relu', 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def map_pooling(self, node):
        return InfoNode('pooling', 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def map_inner_product(self, node):
        return MaybeActivated(node)('fc', 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def map_softmax(self, node):
        return InfoNode('softmax', 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def map_lrn(self, node):
        return InfoNode('lrn', 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def map_concat(self, node):
        return InfoNode('concat', 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def map_dropout(self, node):
        return InfoNode('dropout', 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def map_batch_norm(self, node):
        return MaybeActivated(node)('batch_normalization', 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def map_eltwise(self, node):
        operations = {0: 'multiply', 1: 'add', 2: 'max'}
        op_code = node.parameters.operation
        try:
            return InfoNode(operations[op_code], 0, 0, 0, 0, 0, 0, 0, 0, 0)
        except KeyError:
            raise CompilerError('Unknown elementwise operation: {}'.format(op_code))

    def commit(self, chains):
        return chains


class InfoEmitter(object):

    def __init__(self, tab=None):
        self.tab = tab or ' ' * 2
        self.prefix = ''
        self.collected_allocations = []
        self.collected_nodes = []
        self.collected_edges = []

    def indent(self):
        self.prefix += self.tab

    def outdent(self):
        self.prefix = self.prefix[:-len(self.tab)]

    def statement(self, s):
        return self.prefix + s + '\n'

    def get_parents(self, node):
        return [parent.name for parent in node.node.parents]

    def emit_node(self, node):
        (allocs, code) = node.emit()
        self.collected_allocations += allocs
        self.collected_nodes += list(map(lambda x: self.statement(str(x)), code))

    def convert_edge(self, edge):
      src = edge[0]
      sink = edge[1]

    def emit(self, name, chains, lookup_nodes):

        for chain in chains:
            for node in chain:
                self.emit_node(node)
                self.collected_edges += [(parent, node.node.name) for parent in self.get_parents(node) if parent not in magic_layers and node.node.name not in magic_layers]

        def convert_edge(e):
          e_src = lookup_nodes.index(e[0])
          e_sink = lookup_nodes.index(e[1])
          return str(e_src) + ' ' + str(e_sink) + '\n'

        s = ''.join(self.collected_allocations)
        s += ''.join(self.collected_nodes)
        s += '\n'
        s += ''.join(map(convert_edge, self.collected_edges))
        return s

    def emit_constraints(self, name, chains, lookup_nodes):

        constraints = []
        for chain in chains:
            for node in chain:
                if node.node.name in lookup_nodes:
                    constraints += [node.emit_constraints()]

        return '\n'.join(constraints)


class InfoTransformer(object):

    def __init__(self, def_path, data_path, verbose=True, phase='test'):
        self.verbose = verbose
        self.phase = phase
        self.load(def_path, data_path, phase)
        self.params = None
        self.source = None
        self.no_nodes = 0
        self.no_params = 7
        self.no_edges = 0


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
            ReLUFuser(allowed_parent_types=[LayerKind.Convolution, LayerKind.InnerProduct,
                                            LayerKind.BatchNorm]),

            # Rename nodes
            NodeRenamer(lambda node: node.name.replace('/', '_')),

            # Split concat operations into balanced binary trees
            ConcatTreeSplitter()
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
            g = self.graph.transformed(transformers)

            mapper = InfoMapper(g)
            chains = mapper.map()
            emitter = InfoEmitter()
            actual_nodes = [node.name for node in g.nodes if node.name not in magic_layers]
            toposource_body = emitter.emit(g.name, chains, actual_nodes)
            self.constraintssource = emitter.emit_constraints(g.name, chains, actual_nodes)
            self.no_nodes = len(actual_nodes)
            self.no_edges = len(emitter.collected_edges)
            toposource_header = str(self.no_nodes) + ' ' +  str(self.no_params) + ' ' + str(self.no_edges) + '\n\n'
            self.toposource = toposource_header + toposource_body
            self.layersource = '\n'.join(actual_nodes)
        return [self.toposource, self.layersource, self.constraintssource]
