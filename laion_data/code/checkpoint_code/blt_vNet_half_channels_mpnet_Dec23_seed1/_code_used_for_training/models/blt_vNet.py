'''
Keras implementation of BLT network
'''

import tensorflow as tf
import os, importlib
local_dir = os.path.dirname(os.path.realpath(__file__))

def localdir_modulespec(module_name, dir_path):
    """This function allows us to import functions from remote folders (e.g., needed when loading the model
    from root/save_dir/.../_code_used_for_training) - used in root/task_helper_fuctions.load_model_from_path
    """
    file_path = dir_path + f"/{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BLT_funcs = localdir_modulespec('blt', local_dir+'/layers')
help_funcs = localdir_modulespec('model_helper_functions', local_dir+'/model_helper')


def blt_vNet(input_tensor, classes, hparams, divide_n_channels=1):
        '''Build the computational graph for the model

        Note that evaluations based on model outputs will reflect instantaneous
        rather than cumulative readouts

        Args:
            input_tensor: Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: int, number of classes to classify images into

        Returns:
            model
        '''

        n_recurrent_steps = hparams['n_recurrent_steps']
        n_timesteps = max(n_recurrent_steps, 1)  # to take care of feedforward case (where timesteps = 0)
        norm_type = hparams['norm_type']
        regularize = hparams['regularize']

        use_bias = True if norm_type == 'no_norm' else False

        layers = [
            BLT_funcs.BLTConvLayer(64//divide_n_channels,   7, 'RCL_0', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(64//divide_n_channels,   7, 'RCL_1', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(128//divide_n_channels,  5, 'RCL_2', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(128//divide_n_channels,  5, 'RCL_3', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(256//divide_n_channels,  3, 'RCL_4', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(256//divide_n_channels,  3, 'RCL_5', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(512//divide_n_channels,  3, 'RCL_6', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(512//divide_n_channels,  3, 'RCL_7', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(1024//divide_n_channels, 1, 'RCL_8', use_bias, stride=1, regularize=regularize),
            BLT_funcs.BLTConvLayer(1024//divide_n_channels, 1, 'RCL_9', use_bias, stride=1, regularize=regularize),
        ]
        readout_dense = tf.keras.layers.Dense(
                                classes, kernel_initializer='glorot_uniform',
                                kernel_regularizer=tf.keras.regularizers.l2(regularize),
                                name='ReadoutDense',
                                dtype='float32')  # for numerical stability when using tf's mixed_precision policy
        pooling_layers = [3, 4, 9]
        do_global_average_pooling = True

        # initialise list for activations and outputs
        n_layers = len(layers)
        activations = [[None for _ in range(n_layers)] for _ in range(n_timesteps)]
        outputs = [None for _ in range(n_timesteps)]

        # build the model
        for t in range(n_timesteps):
            for n, layer in enumerate(layers):

                # get the bottom-up input
                if n == 0:
                    b_input = input_tensor if t == 0 else None
                else:
                    if n in pooling_layers:
                        # pool b_input for all layers apart from input
                        b_input = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name=f'MaxPool_Layer_{n}_Time_{t}')(activations[t][n-1])
                    else:
                        b_input = activations[t][n-1]

                # get the lateral input
                l_input = activations[t-1][n] if t >= 1 else None
                
                if n < n_layers - 1:
                    # get the top-down input
                    t_input = activations[t-2][n+1] if t >= 2 else None
                    # BLT convolution
                    x_tn = layer(b_input, l_input, t_input)
                else:
                    # BL convolution
                    x_tn = layer(b_input, l_input)

                # normalization & activation (order depends on hparams['norm-first']
                activations[t][n] = help_funcs.parse_normalization_and_activation(x_tn, n, t, hparams)

            if 'finetune' not in hparams['model_name']:
                # if finetuning, only return the last layer's last timestep activities instead of the readout,
                # hence we skip this part where we add the readout layers
                if do_global_average_pooling:
                    x = tf.keras.layers.GlobalAvgPool2D(name=f'GlobalAvgPool_Time_{t}')(activations[t][-1])
                else:
                    x = activations[t][-1]

                x = readout_dense(x)
            
                outputs[t] = help_funcs.network_output(x, hparams, t)

        if 'finetune' in hparams['model_name']:
            # if finetuning, only return the last layer's last timestep activities instead of the readout
            if do_global_average_pooling:
                    outputs = tf.keras.layers.GlobalAvgPool2D(name=f'GlobalAvgPool_Time_{n_timesteps-1}')(activations[-1][-1])
            else:
                outputs = activations[-1][-1]

        # create Keras model and return
        model = tf.keras.Model(
            inputs=input_tensor,
            outputs=outputs,
            name='blt_vNet')

        return model
