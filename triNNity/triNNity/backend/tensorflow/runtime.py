runtime_header = '''
import argparse
import numpy as np
import tensorflow as tf
import os.path as osp

DEFAULT_PADDING = 'SAME'

class DataSpec(object):

    def __init__(self,
                 batch_size,
                 scale_size,
                 crop_size,
                 isotropic,
                 channels=3,
                 mean=None,
                 bgr=True):
        # The recommended batch size for this model
        self.batch_size = batch_size
        # The image should be scaled to this size first during preprocessing
        self.scale_size = scale_size
        # Whether the model expects the rescaling to be isotropic
        self.isotropic = isotropic
        # A square crop of this dimension is expected by this model
        self.crop_size = crop_size
        # The number of channels in the input image expected by this model
        self.channels = channels
        # The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
        # The values below are ordered BGR, as many Caffe models are trained in this order.
        # Some of the earlier models (like AlexNet) used a spatial three-channeled mean.
        # However, using just the per-channel mean values instead doesn't affect things too much.
        self.mean = mean if mean is not None else np.array([104., 117., 124.])
        # Whether this model expects images to be in BGR order
        self.expects_bgr = True

def layer(op):

    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        layer_output = op(self, layer_input, *args, **kwargs)
        self.layers[name] = layer_output
        self.feed(layer_output)
        return self

    return layer_decorated

class Network(object):

    def __init__(self, inputs, trainable=True):
        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                output = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                output = tf.concat(3, output_groups)
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name=name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

def validate(net, model_path, image_producer, top_k=5):
    input_node = net.inputs['data']
    label_node = tf.placeholder(tf.int32)
    probs = net.get_output()
    top_k_op = tf.nn.in_top_k(probs, label_node, top_k)
    count = 0
    correct = 0
    total = len(image_producer)

    with tf.Session() as sesh:
        coordinator = tf.train.Coordinator()
        net.load(data_path=model_path, session=sesh)
        threads = image_producer.start(session=sesh, coordinator=coordinator)
        for (labels, images) in image_producer.batches(sesh):
            correct += np.sum(sesh.run(top_k_op,
                                       feed_dict={input_node: images,
                                                  label_node: labels}))
            count += len(labels)
            cur_accuracy = float(correct) * 100 / count
            print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)
    print('Top {} Accuracy: {}'.format(top_k, float(correct) / total))

def process_image(img, scale, isotropic, crop, mean):
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.pack([scale, scale])
    img = tf.image.resize_images(img, new_shape[0], new_shape[1])
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    # See: https://github.com/tensorflow/tensorflow/issues/521
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.pack([offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean

class ImageProducer(object):

    def __init__(self, image_paths, data_spec, num_concurrent=4, batch_size=None, labels=None):
        # The data specifications describe how to process the image
        self.data_spec = data_spec
        # A list of full image paths
        self.image_paths = image_paths
        # An optional list of labels corresponding to each image path
        self.labels = labels
        # A boolean flag per image indicating whether its a JPEG or PNG
        self.extension_mask = self.create_extension_mask(self.image_paths)
        # Create the loading and processing operations
        self.setup(batch_size=batch_size, num_concurrent=num_concurrent)

    def setup(self, batch_size, num_concurrent):
        # Validate the batch size
        num_images = len(self.image_paths)
        batch_size = min(num_images, batch_size or self.data_spec.batch_size)
        if num_images % batch_size != 0:
            raise ValueError(
                'The total number of images ({}) must be divisible by the batch size ({}).'.format(
                    num_images, batch_size))
        self.num_batches = num_images / batch_size

        # Create a queue that will contain image paths (and their indices and extension indicator)
        self.path_queue = tf.FIFOQueue(capacity=num_images,
                                       dtypes=[tf.int32, tf.bool, tf.string],
                                       name='path_queue')

        # Enqueue all image paths, along with their indices
        indices = tf.range(num_images)
        self.enqueue_paths_op = self.path_queue.enqueue_many([indices, self.extension_mask,
                                                              self.image_paths])
        # Close the path queue (no more additions)
        self.close_path_queue_op = self.path_queue.close()

        # Create an operation that dequeues a single path and returns a processed image
        (idx, processed_image) = self.process()

        # Create a queue that will contain the processed images (and their indices)
        image_shape = (self.data_spec.crop_size, self.data_spec.crop_size, self.data_spec.channels)
        processed_queue = tf.FIFOQueue(capacity=int(np.ceil(num_images / float(num_concurrent))),
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), image_shape],
                                       name='processed_queue')

        # Enqueue the processed image and path
        enqueue_processed_op = processed_queue.enqueue([idx, processed_image])

        # Create a dequeue op that fetches a batch of processed images off the queue
        self.dequeue_op = processed_queue.dequeue_many(batch_size)

        # Create a queue runner to perform the processing operations in parallel
        num_concurrent = min(num_concurrent, num_images)
        self.queue_runner = tf.train.QueueRunner(processed_queue,
                                                 [enqueue_processed_op] * num_concurrent)

    def start(self, session, coordinator, num_concurrent=4):
        # Queue all paths
        session.run(self.enqueue_paths_op)
        # Close the path queue
        session.run(self.close_path_queue_op)
        # Start the queue runner and return the created threads
        return self.queue_runner.create_threads(session, coord=coordinator, start=True)

    def get(self, session):
        (indices, images) = session.run(self.dequeue_op)
        if self.labels is not None:
            labels = [self.labels[idx] for idx in indices]
            return (labels, images)
        return (indices, images)

    def batches(self, session):
        for _ in xrange(self.num_batches):
            yield self.get(session=session)

    def load_image(self, image_path, is_jpeg):
        # Read the file
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.cond(
            is_jpeg,
            lambda: tf.image.decode_jpeg(file_data, channels=self.data_spec.channels),
            lambda: tf.image.decode_png(file_data, channels=self.data_spec.channels))
        if self.data_spec.expects_bgr:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels.
            img = tf.reverse(img, [False, False, True])
        return img

    def process(self):
        # Dequeue a single image path
        idx, is_jpeg, image_path = self.path_queue.dequeue()
        # Load the image
        img = self.load_image(image_path, is_jpeg)
        # Process the image
        processed_img = process_image(img=img,
                                      scale=self.data_spec.scale_size,
                                      isotropic=self.data_spec.isotropic,
                                      crop=self.data_spec.crop_size,
                                      mean=self.data_spec.mean)
        # Return the processed image, along with its index
        return (idx, processed_img)

    @staticmethod
    def create_extension_mask(paths):

        def is_jpeg(path):
            extension = osp.splitext(path)[-1].lower()
            if extension in ('.jpg', '.jpeg'):
                return True
            if extension != '.png':
                raise ValueError('Unsupported image format: {}'.format(extension))
            return False

        return [is_jpeg(p) for p in paths]

    def __len__(self):
        return len(self.image_paths)

class ListImageProducer(ImageProducer):

    def __init__(self, val_path, data_path, data_spec):
        # Read in the ground truth labels for the validation set
        # The get_ilsvrc_aux.sh in Caffe's data/ilsvrc12 folder can fetch a copy of val.txt
        gt_lines = open(val_path).readlines()
        gt_pairs = [line.split() for line in gt_lines]
        # Get the full image paths
        # You will need a copy of the ImageNet validation set for this.
        image_paths = [osp.join(data_path, p[0]) for p in gt_pairs]
        # The corresponding ground truth labels
        labels = np.array([int(p[1]) for p in gt_pairs])
        # Initialize base
        super(ImageNetProducer, self).__init__(image_paths=image_paths,
                                               data_spec=data_spec,
                                               labels=labels)
'''

runtime_main = '''

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('weights', help='Path to the converted weights (.npy)')
  parser.add_argument('image_list', help='Path to validation set ground truth (.txt)')
  parser.add_argument('data_dir', help='Path to validation set images directory')
  parser.add_argument('top_k', help='Value to use for top-N')
  args = parser.parse_args()

  spec = DataSpec({}, {}, {}, {}, {}, {}, {})

  image_source = ListImageProducer(args.image_list, args.data_dir, spec)

  input_node = tf.placeholder(tf.float32,
                              shape=(None, spec.crop_size, spec.crop_size, spec.channels))

  net = {}({'data': input_node})

  validate(net, args.weights, image_source, args.top_k)
'''

class TensorFlowRuntime(object):

    def __init__(self, output):
        self.output = output

    def generate(self, code, net_name):
      self.output.write(runtime_header)
      self.output.write(code[0])
      self.output.write(runtime_main.format(net_name))
