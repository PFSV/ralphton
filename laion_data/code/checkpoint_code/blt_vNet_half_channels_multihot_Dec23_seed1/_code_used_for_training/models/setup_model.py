'''Selection of models to be used in training
'''

import tensorflow as tf
from tensorflow.keras.layers import Input
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


def blt_vNet_model_fn(input_shape, n_classes, hparams):
    input_layer = Input(batch_shape=input_shape)
    module = localdir_modulespec('blt_vNet', local_dir)
    model = module.blt_vNet(input_layer, n_classes, hparams, divide_n_channels=1)
    return model


def blt_vNet_half_channels_model_fn(input_shape, n_classes, hparams):
    input_layer = Input(batch_shape=input_shape)
    module = localdir_modulespec('blt_vNet', local_dir)
    model = module.blt_vNet(input_layer, n_classes, hparams, divide_n_channels=2)
    return model


def simclr_blt_vNet_model_fn(input_shape, n_classes, hparams):
    input_layer = Input(batch_shape=input_shape)
    bltsimclr_module = localdir_modulespec('blt_vNet_simCLR', local_dir)
    simclrhelper_module = localdir_modulespec('model_helper/simclr_helper_functions', local_dir)
    augmenter = simclrhelper_module.get_augmenter(**simclrhelper_module.CONTRASTIVE_AUGMENTATION, hparams=hparams) 
    simclr_model = bltsimclr_module.simclr_bltvNet(augmenter, input_layer, hparams, divide_n_channels=1)
    
    return simclr_model


def simclr_blt_vNet_half_channels_model_fn(input_shape, n_classes, hparams):
    input_layer = Input(batch_shape=input_shape)
    bltsimclr_module = localdir_modulespec('blt_vNet_simCLR', local_dir)
    simclrhelper_module = localdir_modulespec('model_helper/simclr_helper_functions', local_dir)
    augmenter = simclrhelper_module.get_augmenter(**simclrhelper_module.CONTRASTIVE_AUGMENTATION, hparams=hparams) 
    simclr_model = bltsimclr_module.simclr_bltvNet(augmenter, input_layer, hparams, divide_n_channels=2)
    
    return simclr_model


def finetuning_model_fn(base_model, input_shape, n_classes, hparams, freeze_base_model=True):
    '''Please not that, for recurrent models, this finetunes based on the last timestep only.
    This is done in blt_vNet.py, where we have a special case when finetuning, and only return
    the last timestep.'''

    if freeze_base_model:
        for layer in base_model.layers:
            layer.trainable = False

    output_activation = None if hparams['model_output_activation'] == 'no_model_output_activation' else hparams['model_output_activation']

    finetuning_model = tf.keras.Sequential([
        tf.keras.Input(batch_shape=input_shape),
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(n_classes, activation=output_activation, name=f'output_time_{hparams["n_recurrent_steps"]-1}'),
    ], name="finetuned_simclr_model")
    return finetuning_model


def simclr_encoder_finetuning_model_fn(simclr_encoder, input_shape, n_classes, hparams, freeze_base_model=True):
    
    if freeze_base_model:
        for layer in simclr_encoder.layers:
            layer.trainable = False

    output_activation = None if hparams['model_output_activation'] == 'no_model_output_activation' else hparams['model_output_activation']
        
    simclr_helper_module = localdir_modulespec('model_helper/simclr_helper_functions', local_dir)
    classification_augmenter = simclr_helper_module.get_augmenter(**simclr_helper_module.CLASSIFICATION_AUGMENTATION, hparams=hparams)  
    finetuning_simclr_model = tf.keras.Sequential([
        tf.keras.Input(batch_shape=input_shape),
        classification_augmenter,
        simclr_encoder,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(n_classes, activation=output_activation, name=f'output_time_{hparams["n_recurrent_steps"]-1}'),
    ], name="finetuned_simclr_model")
    return finetuning_simclr_model


def get_model_function(model_name):
    if model_name == 'blt_vNet':
        return blt_vNet_model_fn
    elif model_name == 'blt_vNet_half_channels':
        return blt_vNet_half_channels_model_fn
    if model_name == 'simclr_blt_vNet':
        return simclr_blt_vNet_model_fn
    elif model_name == 'simclr_blt_vNet_half_channels':
        return simclr_blt_vNet_half_channels_model_fn
    elif model_name in ['blt_vNet_finetune', 'blt_vNet_half_channels_finetune']:
        return finetuning_model_fn
    elif model_name in ['simclr_blt_vNet_finetune', 'simclr_blt_vNet_half_channels_finetune']:
        return simclr_encoder_finetuning_model_fn
    else:
        raise ValueError('Model not available: {}'.format(model_name))
