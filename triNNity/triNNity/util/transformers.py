'''
A collection of graph transforms.

A transformer is a callable that accepts a graph and returns a transformed version.
'''

import numpy as np

from triNNity.util.caffe import get_caffe_resolver, has_pycaffe
from triNNity.util.errors import CompilerError, print_stderr
from triNNity.frontend.layers import LayerKind
from triNNity.frontend.graph import IRNode
from triNNity.frontend.shapes import TensorShape

class DataInjector(object):
    '''
    Associates parameters loaded from a .caffemodel file with their corresponding nodes.
    '''

    def __init__(self, def_path, data_path):
        # The .prototxt file defining the graph
        self.def_path = def_path
        # The .caffemodel file containing the learned parameters
        self.data_path = data_path
        # Set to true if the fallback protocol-buffer based backend was used
        self.did_use_pb = False
        # A list containing (layer name, parameters) tuples
        self.params = None
        # Load the parameters
        self.load()

    def load(self):
        if has_pycaffe():
            self.load_using_caffe()
        else:
            self.load_using_pb()

    def load_using_caffe(self):
        caffe = get_caffe_resolver().caffe
        net = caffe.Net(self.def_path, self.data_path, caffe.TEST)
        data = lambda blob: blob.data
        self.params = [(k, list(map(data, v))) for k, v in net.params.items()]

    def load_using_pb(self):
        data = get_caffe_resolver().NetParameter()
        data.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.normalize_pb_data(layer))
        layers = data.layers or data.layer
        self.params = [pair(layer) for layer in layers if layer.blobs]
        self.did_use_pb = True

    def normalize_pb_data(self, layer):
        transformed = []
        for blob in layer.blobs:
            if len(blob.shape.dim):
                dims = blob.shape.dim
                c_o, c_i, h, w = map(int, [1] * (4 - len(dims)) + list(dims))
            else:
                c_o = blob.num
                c_i = blob.channels
                h = blob.height
                w = blob.width
            data = np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w)
            transformed.append(data)
        return transformed

    def adjust_parameters(self, node, data):
        if not self.did_use_pb:
            return data
        # When using the protobuf-backend, each parameter initially has four dimensions.
        # In certain cases (like FC layers), we want to eliminate the singleton dimensions.
        # This implementation takes care of the common cases. However, it does leave the
        # potential for future issues.
        # The Caffe-backend does not suffer from this problem.
        data = list(data)
        squeeze_indices = [1]  # Squeeze biases.
        if node.kind == LayerKind.InnerProduct:
            squeeze_indices.append(0)  # Squeeze FC.
        for idx in squeeze_indices:
            data[idx] = np.squeeze(data[idx])
        return data

    def __call__(self, graph):
        for layer_name, data in self.params:
            if layer_name in graph:
                node = graph.get_node(layer_name)
                node.data = self.adjust_parameters(node, data)
            else:
                print_stderr('Ignoring parameters for non-existent layer: %s' % layer_name)
        return graph


class DataReshaper(object):

    def __init__(self, mapping, replace=True):
        # A dictionary mapping LayerKind to the transposed order.
        self.mapping = mapping
        # The node kinds eligible for reshaping
        self.reshaped_node_types = self.mapping.keys()
        # If true, the reshaped data will replace the old one.
        # Otherwise, it's set to the reshaped_data attribute.
        self.replace = replace

    def has_spatial_parent(self, node):
        try:
            parent = node.parents[0]
            s = parent.output_shape
            return s.height > 1 or s.width > 1
        except CompilerError:
            return False

    def map(self, node_kind):
        try:
            return self.mapping[node_kind]
        except KeyError:
            raise CompilerError('Ordering not found for node kind: {}'.format(node_kind))

    def __call__(self, graph):
        for node in graph.nodes:
            if node.data is None:
                continue
            if node.kind not in self.reshaped_node_types:
                # Check for 2+ dimensional data
                if any(len(tensor.shape) > 1 for tensor in node.data):
                    print_stderr('Warning: parameters not reshaped for node: {}'.format(node))
                continue
            transpose_order = self.map(node.kind)
            if len(list(node.data)) == 0:
                raise CompilerError('Missing weights for node: {}'.format(node))
            weights = list(node.data)[0]
            if (node.kind == LayerKind.InnerProduct) and self.has_spatial_parent(node):
                # The FC layer connected to the spatial layer needs to be
                # re-wired to match the new spatial ordering.
                in_shape = node.get_only_parent().output_shape
                fc_shape = weights.shape
                output_channels = fc_shape[0]
                weights = weights.reshape((output_channels, in_shape.channels, in_shape.height,
                                           in_shape.width))
                weights = weights.transpose(self.map(LayerKind.Convolution))
                node.reshaped_data = weights.reshape(fc_shape[transpose_order[0]],
                                                     fc_shape[transpose_order[1]])
            else:
                node.reshaped_data = weights.transpose(transpose_order)

        if self.replace:
            for node in graph.nodes:
                if hasattr(node, 'reshaped_data'):
                    # Set the weights
                    list(node.data)[0] = node.reshaped_data
                    del node.reshaped_data
        return graph

def chunks_of(size, seq):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class ConcatTreeSplitter(object):
    '''
    A helper for splitting concat layers into trees of pairwise concatenations
    '''

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, graph):
        new_subgraphs = []
        kill_nodes = []
        finished_nodes = []
        for node in graph.nodes:
            if node.kind == LayerKind.Concat and len(node.parents) > 2:
                kill_nodes.append(node.name)
                unique_id = 0

                inputs = node.parents
                outputs = node.children

                temp_inputs = []
                for maybe_pair in list(chunks_of(2, inputs)):
                    if len(maybe_pair) == 2:
                        new_node = IRNode(node.name+'_split_'+str(unique_id), LayerKind.Concat)
                        new_node.layer = node.layer

                        for x in maybe_pair[0].children:
                            maybe_pair[0].del_child(x)
                            if self.verbose:
                                print("Removing edge " + maybe_pair[0].name + " -> " + x.name)

                        for x in maybe_pair[1].children:
                            maybe_pair[1].del_child(x)
                            if self.verbose:
                                print("Removing edge " + maybe_pair[1].name + " -> " + x.name)

                        new_node.add_parent(maybe_pair[0])
                        if self.verbose:
                            print("Adding edge " + maybe_pair[0].name + " -> " + new_node.name)

                        new_node.add_parent(maybe_pair[1])
                        if self.verbose:
                            print("Adding edge " + maybe_pair[1].name + " -> " + new_node.name)

                        c_in_l = maybe_pair[0].output_shape[1]
                        c_in_r = maybe_pair[1].output_shape[1]
                        h_o = maybe_pair[0].output_shape[2]
                        w_o = maybe_pair[0].output_shape[3]
                        new_node.output_shape = TensorShape(node.output_shape[0], c_in_l+c_in_r, h_o, w_o)
                        temp_inputs.append(new_node)
                        unique_id += 1
                        finished_nodes.append(new_node)
                    else:
                        temp_inputs.append(maybe_pair[0])

                    inputs = temp_inputs
                    temp_inputs = []

                    if len(inputs) == 1:
                        # this node is the new root
                        root_node = inputs[0]
                        for x in outputs:
                            x.del_parent(node)
                            if self.verbose:
                                print("Removing edge " + node.name + " -> " + x.name)
                            x.add_parent(root_node)
                            if self.verbose:
                                print("Adding edge " + root_node.name + " -> " + x.name)
                        finished_nodes.append(root_node)

                new_subgraphs += finished_nodes

                for x in node.parents:
                    x.del_child(node)
                    if self.verbose:
                        print("Removing edge " + x.name + " -> " + node.name)

        graph.deadnames += kill_nodes
        newGraph = graph.replaced([n for n in graph.nodes+new_subgraphs if n.name not in graph.deadnames])
        newGraph.deadnames = graph.deadnames
        return newGraph


class SubNodeFuser(object):
    '''
    An abstract helper for merging a single-child with its single-parent.
    '''

    def __call__(self, graph):
        nodes = graph.nodes
        fused_nodes = []
        for node in nodes:
            if len(node.parents) != 1:
                # We're only fusing nodes with single parents
                continue
            parent = node.get_only_parent()
            if len(parent.children) != 1:
                # We can only fuse a node if its parent's
                # value isn't used by any other node.
                continue
            if not self.is_eligible_pair(parent, node):
                continue
            # Rewrite the fused node's children to its parent.
            for child in node.children:
                child.parents.remove(node)
                parent.add_child(child)
            # Disconnect the fused node from the graph.
            parent.children.remove(node)
            fused_nodes.append(node)
            # Let the sub-class merge the fused node in any arbitrary way.
            self.merge(parent, node)
        transformed_nodes = [node for node in nodes if node not in fused_nodes]
        return graph.replaced(transformed_nodes)

    def is_eligible_pair(self, parent, child):
        '''Returns true if this parent/child pair is eligible for fusion.'''
        raise NotImplementedError('Must be implemented by subclass.')

    def merge(self, parent, child):
        '''Merge the child node into the parent.'''
        raise NotImplementedError('Must be implemented by subclass')


class ReLUFuser(SubNodeFuser):
    '''
    Fuses rectified linear units with their parent nodes.
    '''

    def __init__(self, allowed_parent_types=None):
        # Fuse ReLUs when the parent node is one of the given types.
        # If None, all node types are eligible.
        self.allowed_parent_types = allowed_parent_types

    def is_eligible_pair(self, parent, child):
        return ((self.allowed_parent_types is None or parent.kind in self.allowed_parent_types) and
                child.kind == LayerKind.ReLU)

    def merge(self, parent, _):
        parent.metadata['relu'] = True


class BatchNormScaleBiasFuser(SubNodeFuser):
    '''
    The original batch normalization paper includes two learned
    parameters: a scaling factor \gamma and a bias \beta.
    Caffe's implementation does not include these two. However, it is commonly
    replicated by adding a scaling+bias layer immidiately after the batch norm.

    This fuser merges the scaling+bias layer with the batch norm.
    '''

    def is_eligible_pair(self, parent, child):
        return (parent.kind == LayerKind.BatchNorm and child.kind == LayerKind.Scale and
                child.parameters.axis == 1 and child.parameters.bias_term == True)

    def merge(self, parent, child):
        parent.scale_bias_node = child


class BatchNormPreprocessor(object):
    '''
    Prescale batch normalization parameters.
    Concatenate gamma (scale) and beta (bias) terms if set.
    '''

    def __call__(self, graph):
        for node in graph.nodes:
            if node.kind != LayerKind.BatchNorm:
                continue
            assert node.data is not None
            assert len(node.data) == 3
            mean, variance, scale = node.data
            # Prescale the stats
            scaling_factor = 1.0 / scale if scale != 0 else 0
            mean *= scaling_factor
            variance *= scaling_factor
            # Replace with the updated values
            node.data = [mean, variance]
            if hasattr(node, 'scale_bias_node'):
                # Include the scale and bias terms
                gamma, beta = node.scale_bias_node.data
                node.data += [gamma, beta]
        return graph


class NodeRenamer(object):
    '''
    Renames nodes in the graph using a given unary function that
    accepts a node and returns its new name.
    '''

    def __init__(self, renamer):
        self.renamer = renamer

    def __call__(self, graph):
        for node in graph.nodes:
            node.name = self.renamer(node)
        return graph


class ParameterNamer(object):
    '''
    Convert layer data arrays to a dictionary mapping parameter names to their values.
    '''

    def __call__(self, graph):
        for node in graph.nodes:
            if node.data is None:
                continue
            if node.kind == LayerKind.InnerProduct:
                names = ('weights',)
                if node.parameters.bias_term:
                    names += ('biases',)
            elif node.kind == LayerKind.Convolution:
                names = ('weights',)
                names += ('biases',)
                names += ('weights_masked',)
                names += ('biases_masked',)
            elif node.kind == LayerKind.BatchNorm:
                names = ('mean', 'variance')
                if len(node.data) == 4:
                    names += ('scale', 'offset')
            else:
                print_stderr('Warning: Unhandled parameters: {}'.format(node.kind))
                continue
            if len(names) > len(list(node.data)):
                raise CompilerError('parameter mismatch in node: {}, expected {} parameters, got {}'.format(node, len(names), len(list(node.data))))
            # Ignore extra data blobs or extra names
            awful_hack = min(len(names), len(list(node.data)))
            node.data = dict(zip(names, list(node.data)[:awful_hack]))
        return graph
