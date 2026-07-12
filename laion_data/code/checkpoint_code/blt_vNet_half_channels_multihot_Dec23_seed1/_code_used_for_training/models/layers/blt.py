import tensorflow as tf


class BLTDenseLayer(object):
    '''BLT recurrent dense layer

    Note that this is NOT A KERAS LAYER but is an object containing Keras layers

    Args:
        n_units: Int, number of units in dense layer
        layer_name: String, prefix for layers in the RCL
        readout: Bool, set to true if this is the readout layer. In this case, there are no top-down or lateral connections.
            (also changes self.get_layer_type() so we can adapt code in the model function)
        '''

    def __init__(self, n_units, layer_name, use_bias, regularize=1e-6, readout=False):

        self.layer_name = layer_name
        self.readout = readout

        # initialise convolutional layers
        if readout:
            # only bottom-up connections in this case
            self.b_dense = tf.keras.layers.Dense(
                n_units, use_bias=use_bias, kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(regularize),
                name=f'{layer_name}_BDense', dtype='float32')  # dtype needed for stability if using mixed_precision
        else:
            self.b_dense = tf.keras.layers.Dense(
                n_units, use_bias=use_bias, kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(regularize),
                name=f'{layer_name}_BDense')

            self.l_dense = tf.keras.layers.Dense(
                n_units, use_bias=use_bias, kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(regularize),
                name=f'{layer_name}_LDense')

            self.t_dense = tf.keras.layers.Dense(
                n_units, use_bias=use_bias, kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(regularize),
                name=f'{layer_name}_TDense')

        # layer for summing convolutions
        self.sum_dense = tf.keras.layers.Lambda(tf.add_n, name=f'{layer_name}_DenseSum')

        # holds the most recent bottom-up conv
        # useful when the bottom-up input does not change, e.g. input image
        self.previous_b_dense = None

    def get_layer_type(self):
        return 'Readout' if self.readout else 'BLTDenseLayer'

    def __call__(self, b_input=None, l_input=None, t_input=None):
        dense_list = []

        if not b_input is None:
            # run bottom-up conv and save result
            dense_list.append(self.b_dense(b_input))
            self.previous_b_dense = dense_list[-1]
        elif not self.previous_b_dense is None:
            # use the most recent bottom-up conv
            dense_list.append(self.previous_b_dense)
        else:
            raise ValueError('b_input must be given on first pass')

        # run lateral conv
        if l_input is not None:
            dense_list.append(self.l_dense(l_input))

        # run top-down conv
        if t_input is not None:
            dense_list.append(self.t_dense(t_input))

        # return element-wise sum of convolutions
        return self.sum_dense(dense_list)


class BLTConvLayer(object):
    '''BLT recurrent convolutional layer

    Note that this is NOT A KERAS LAYER but is an object containing Keras layers

    Args:
        filters: Int, number of output filters in convolutions
        kernel_size: Int or tuple/list of 2 integers, specifying the height and
            width of the 2D convolution window. Can be a single integer to
            specify the same value for all spatial dimensions.
        layer_name: String, prefix for layers in the RCL
        '''

    def __init__(self, filters, kernel_size, layer_name, use_bias, stride=1, regularize=1e-6, unpool_t_input=False, l_flag=True, t_flag=True, lt_interact=0):

        # initialise convolutional layers
        self.l_flag = l_flag
        self.t_flag = t_flag
        self.lt_interact = lt_interact  # if 0: additive feedback, if 1: multiplicative

        self.b_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=stride, padding='same', use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(regularize),
            name=f'{layer_name}_BConv')

        if l_flag:
            self.l_conv = tf.keras.layers.Conv2D(
                filters, kernel_size, padding='same', use_bias=use_bias,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(regularize),
                name=f'{layer_name}_LConv')
        else:
            self.l_conv = None

        if t_flag:
            if unpool_t_input:
                self.t_dense = tf.keras.layers.Dense(
                    filters, use_bias=use_bias, kernel_initializer='glorot_uniform',
                    kernel_regularizer=tf.keras.regularizers.l2(regularize),
                    name=f'{layer_name}_TDense')
            else:
                self.t_conv = tf.keras.layers.Conv2D(
                    filters, kernel_size, padding='same', use_bias=use_bias,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=tf.keras.regularizers.l2(regularize),
                    name=f'{layer_name}_TConv')
        else:
            self.t_conv = None

        # layer for summing convolutions
        if self.lt_interact == 0:
            self.sum_convs = tf.keras.layers.Lambda(
                tf.add_n, name='{}_ConvSum'.format(layer_name))
        elif self.lt_interact == 1:
            def multiplier_lambda(x):
                return tf.multiply(x[0],x[1])
            self.multiply_convs = tf.keras.layers.Lambda(
                multiplier_lambda, name='{}_ConvMult'.format(layer_name))
        else:
            raise ValueError('lt_interact must be 0 (additive feedback) or 1 (multiplicative feedback).')

        # layer for resizing T conv is initialised when first called
        self.resize_layer = None
        self.tile_layer = None
        self.layer_name = layer_name
        self.unpool_t_input = unpool_t_input

        # holds the most recent bottom-up conv
        # useful when the bottom-up input does not change, e.g. input image
        self.previous_b_conv = None

    def get_layer_type(self):
        return 'BLTConvLayer'

    def __call__(self, b_input=None, l_input=None, t_input=None):
        conv_list = []

        if not b_input is None:
            # run bottom-up conv and save result
            conv_list.append(self.b_conv(b_input))
            self.previous_b_conv = conv_list[-1]
        elif not self.previous_b_conv is None:
            # use the most recent bottom-up conv
            conv_list.append(self.previous_b_conv)
        else:
            raise ValueError('b_input must be given on first pass')

        # run lateral conv
        if l_input is not None:
            if self.l_flag:
                conv_list.append(self.l_conv(l_input))

        # run top-down conv
        if t_input is not None:
            if self.t_flag:
                if self.unpool_t_input:
                    t_out_pooled = self.t_dense(t_input)
                    tiled_t_out = self.tile_t_dense(t_out_pooled, self.b_conv.output_shape)
                    conv_list.append(tiled_t_out)
                else:
                    t_out = self.t_conv(t_input)
                    resized_t_out = self.resize_t_conv(t_out, self.b_conv.output_shape)
                    conv_list.append(resized_t_out)

        # return element-wise sum of ff, l and t convolutions, or element-wise multiplication
        if self.lt_interact == 0:
            out_h = self.sum_convs(conv_list)
        else:
            if len(conv_list) == 1:
                out_h = conv_list[0]
            elif len(conv_list) == 2:
                out_h = self.multiply_convs([1.+conv_list[1], conv_list[0]])
            else:
                out_h = self.multiply_convs([1.+(conv_list[1]+conv_list[2]), conv_list[0]])

        return out_h

    def resize_t_conv(self, x, output_shape):
        '''Resizes T conv and initialises if required
        '''
        if self.resize_layer == None:
            # initalise resize layer
            self.resize_layer = tf.keras.layers.Lambda(
                resize_tensor_for_tconv,
                arguments={'output_shape': output_shape},
                name='{}_ResizeTConv'.format(self.layer_name))

        return self.resize_layer(x)

    def tile_t_dense(self, x, output_shape):
        '''Tile T dense and initialises if required
        '''
        x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
        if self.tile_layer == None:
            # initalise resize layer
            self.tile_layer = tf.keras.layers.Lambda(
                tf.tile,
                arguments={'multiples': (1,)+output_shape[1:3]+(1,)},
                name='{}_TileTDense'.format(self.layer_name))

        return self.tile_layer(x)

def resize_tensor_for_tconv(x, output_shape):
    '''Resize a layer to the target output shape
    '''

    data_format = tf.keras.backend.image_data_format()

    if data_format == 'channels_first':
        # transform to channels last
        x = tf.transpose(x, [0, 2, 3, 1])
        new_height, new_width = output_shape[-2:]
    else:
        new_height, new_width = output_shape[-3:-1]

    x = tf.image.resize(x, size=(new_height, new_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if data_format == 'channels_first':
        # transform back to channels_first
        x = tf.transpose(x, [0, 3, 1, 2])

    return x
