import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
import os, h5py, pickle, importlib, math, glob
from shutil import copyfile
from models.model_helper.model_helper_functions import save_full_model


def get_model(hparams, n_classes, strategy=None, saved_model_path=None):
    '''get model from model function
    '''

    # get input shape
    input_shape = [hparams['batch_size'], hparams['image_size'], hparams['image_size'], 3]

    # adjust batch_size depending on paralellization strategy
    if strategy is not None:
        input_shape[0] = int(input_shape[0] / int(strategy.num_replicas_in_sync))

    # if saved model path is given as input, then we use the model function that was saved with the model instead of the
    # model function in models.setup_model. This is done to ensure that the exact same model function is used for
    # training and testing
    if saved_model_path is not None:
        file_path = saved_model_path + '/_code_used_for_training/models'
        module_name = 'setup_model'
        setup_model = localdir_modulespec(module_name, file_path)
    else:
        from models import setup_model

    # get model function and call it to get the keras model
    # If you get an error here, it is probably because the model you call does not support some of the options you
    # passed, such as 'norm_type'. Look into the ./models folder.
    model_function = setup_model.get_model_function(hparams['model_name'])
    
    if 'finetune' in hparams['model_name'].lower():
        if 'simclr' in hparams['model_name'].lower():
            base_model_function = setup_model.get_model_function(hparams['model_name'].replace('_finetune',''))
            base_model = base_model_function(input_shape, n_classes, hparams)
            base_model_encoder = base_model.encoder
            base_model_encoder.build(input_shape)
            ckpt_name = f'ckpt_ep{hparams["model_to_finetune_epoch"]}.h5' if hparams["model_to_finetune_epoch"] > 0 else 'model_weights_init.h5'
            weights_to_load = os.path.join(hparams['model_to_finetune_savedir'], 'training_checkpoints', ckpt_name)
            base_model_encoder.load_weights(weights_to_load)
            net = model_function(base_model_encoder, input_shape, n_classes, hparams, freeze_base_model=True)
        else:
            base_model_function = setup_model.get_model_function(hparams['model_name'].replace('_finetune',''))
            base_model = base_model_function(input_shape, n_classes, hparams)
            ckpt_name = f'ckpt_ep{hparams["model_to_finetune_epoch"]}.h5' if hparams["model_to_finetune_epoch"] > 0 else 'model_weights_init.h5'
            weights_to_load = os.path.join(hparams['model_to_finetune_savedir'], 'training_checkpoints', ckpt_name)
            # The by_name argument only loads weights to layers with identical names. This is good because it allows us to ignore
            # the readout layer. The way it works is that the blt_vNet model function has a special case when "finetune" is in the 
            # model name, and it does not add the readout layer. So when we load the weights with by_name=True, the readout layer weights
            # are ignored. If we did not use by_name=True, then we would get an error because there are more weights to load than layers.
            base_model.load_weights(weights_to_load, by_name=True)  
            net = model_function(base_model, input_shape, n_classes, hparams, freeze_base_model=True)
    else:
        net = model_function(input_shape, n_classes, hparams)

    return net


def get_losses_and_metrics(network_output_layers, hparams):
    '''Gets loss, metric and loss_weights dict with optional RDL objective
    '''

    if 'simclr' in hparams['model_name'].lower() and not 'finetune' in hparams['model_name'].lower():
        import keras_cv  # importing here so the install is not required for normal models
        print('Using a self-supervised simCLR model.')
        from models.model_helper.simclr_helper_functions import SimCLRLossMetric
        if hparams['simclr_temperature'] == 'none':
            raise Exception('You have "simclr" in hparams["model_name"]. This means that '
                            'you are using a simCLR model. But you have not specified a temperature. '
                            'Please specify a temperature in hparams["simclr_temperature"]')
        loss_dict = keras_cv.losses.SimCLRLoss(temperature=hparams['simclr_temperature'])
        metric_dict = SimCLRLossMetric()
        # These are not dicts, but it still works.
        return loss_dict, metric_dict
    
    else:
        # initialise dictionaries with categorisation losses and metrics
        loss_dict, metric_dict = {}, {}

        def rescaled_cosine_loss():
            # scale the native tf loss in [-1,0] to [0,2]
            def loss(y_true, y_pred):
                return (tf.keras.losses.cosine_similarity(y_true, y_pred) + 1)
            return loss

        for layer in network_output_layers:
            if hparams['embedding_target'] and (hparams['embedding_loss'].lower() == 'cosine'):
                # apply cosine distance loss between predicted and label embeddings. otherwise apply normal classification loss.
                loss_dict[layer] = rescaled_cosine_loss()
                metric_dict[layer] = tf.keras.metrics.CosineSimilarity()
            elif hparams['embedding_target'] and (hparams['embedding_loss'].lower() == 'mse'):
                # apply MSE distance loss between predicted and label embeddings. otherwise apply normal classification loss.
                loss_dict[layer] = tf.keras.losses.MeanSquaredError()
                metric_dict[layer] = tf.keras.metrics.MeanSquaredError()
            else:
                # iterate through output layers applying category objective
                loss_dict[layer] = tf.keras.losses.CategoricalCrossentropy()
                metric_dict[layer] = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy()]

    return loss_dict, metric_dict


def compile_model(net, hparams, loss_dict, metric_dict):
    # specifies optimiser, metrics and loss function for training

    if hparams['optimizer'] == 'adam':
        if hparams['clip_norm'] is not None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], clipnorm=hparams['clip_norm'],
                                                 epsilon=hparams['optim_epsilon'])
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], epsilon=hparams['optim_epsilon'])
    elif hparams['optimizer'].lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'], momentum=hparams['sgd_momentum'],
                                            nesterov=hparams['sgd_nesterov'])
    else:
        raise Exception(f'Optimizer {hparams["optimizer"]} not implemented')
    
    if 'simclr' in hparams['model_name'].lower() and not 'finetune' in hparams['model_name'].lower():
        net.compile(encoder_loss=loss_dict, probe_metrics=metric_dict, encoder_optimizer=optimizer)  # not tested
        for projector in net.projectors:
            projector.build(net.encoder.output_shape) # Need to build the projector shape
    else:
        net.compile(loss=loss_dict, metrics=metric_dict, optimizer=optimizer)

    return net


def get_activities_model(model, model_name):
    """Get a model that returns activities"""

    print('Using get_activities_model from task_helper_functions.py')
    # make keras model to collect layer activities
    n_layers = len(model.layers)
    readout_layers = []
    for layer_id in range(n_layers):
        for this_layer in model.layers:
            if 'sheet_{}'.format(layer_id) in this_layer.name.lower():
                readout_layers.append(this_layer.output)

    activities_model = tf.keras.Model(inputs=model.input, outputs=readout_layers, name=f'{model_name}_activities')

    for layer in activities_model.layers:
        layer.trainable = False

    return activities_model


def make_saving_name(hparams):
    saving_name = hparams['model_name']
    saving_name += f'{hparams["model_name_suffix"]}'
    return saving_name


def save_code_and_params(hparams, model_savedir):

    # save hparams and used code for future reference
    with open(f'{model_savedir}/hparams.txt', 'w') as f:
        [f.write(f'{k}: {v}\n') for k, v in hparams.items()]
    with open(f'{model_savedir}/hparams.pickle', 'wb') as f:
        pickle.dump(hparams, f, protocol=pickle.HIGHEST_PROTOCOL)

    code_saving_path_root = f'{model_savedir}/_code_used_for_training'
    os.makedirs(f'{code_saving_path_root}', exist_ok=True)
    copyfile("task.py", f'{code_saving_path_root}/task.py')
    copyfile("task_helper_functions.py", f'{code_saving_path_root}/task.py')
    copyfile("run_n_epochs.py", f'{code_saving_path_root}/run_n_epochs.py')

    code_saving_path_models = f'{code_saving_path_root}/models'
    os.makedirs(code_saving_path_models, exist_ok=True)
    os.makedirs(f'{code_saving_path_models}/layers', exist_ok=True)
    os.makedirs(f'{code_saving_path_models}/model_helper', exist_ok=True)

    model_file = get_model_file_from_name(hparams['model_name'].replace('_finetune',''))
    copyfile(f"./models/{model_file}.py",f'{code_saving_path_models}/{model_file}.py')
    copyfile(f"./models/setup_model.py", f'{code_saving_path_models}/setup_model.py')
    copyfile(f"./models/model_helper/model_helper_functions.py", f'{code_saving_path_models}/model_helper/model_helper_functions.py')
    if 'tnn' in hparams['model_name'].lower():
        copyfile(f"./models/layers/local.py", f'{code_saving_path_models}/layers/local.py')
        copyfile(f"./models/model_helper/tnn_spatial_loss_helper.py", f'{code_saving_path_models}/model_helper/tnn_spatial_loss_helper.py')
        copyfile(f"./models/model_helper/tnn_helper_functions.py", f'{code_saving_path_models}/model_helper/tnn_helper_functions.py')
    if 'blt' in hparams['model_name'].lower():
        copyfile(f"./models/layers/blt.py", f'{code_saving_path_models}/layers/blt.py')
    if 'simclr' in hparams['model_name'].lower():
        copyfile(f"./models/model_helper/simclr_helper_functions.py", f'{code_saving_path_models}/model_helper/simclr_helper_functions.py')

    code_saving_path_dataset = f'{code_saving_path_root}/dataset_loader'
    os.makedirs(code_saving_path_dataset, exist_ok=True)
    copyfile("./dataset_loader/make_tf_dataset.py", f'{code_saving_path_dataset}/make_tf_dataset.py')
    copyfile("./dataset_loader/tf_dataset_helper_functions.py",
                f'{code_saving_path_dataset}/tf_dataset_helper_functions.py')


def get_model_file_from_name(model_name):
    '''Most of the time, the model file is the same as the model name, but you can take care of exceptions here'''
    if model_name == 'tnn_conv_control':
        model_name = 'tnn'
    if '_half_channels' in model_name:
        model_name = model_name.replace("_half_channels", "")
    if model_name == 'simclr_blt_vNet':
        model_name = 'blt_vNet_simCLR'

    return model_name


def load_model_from_path(saved_model_path, epoch_to_load, n_classes=None, test_mode=False, print_summary=False, hparams=None):

    if hparams is None:
        with open(f'{saved_model_path}/hparams.pickle', 'rb') as f:
            hparams = pickle.load(f)

    if n_classes is None:
        n_classes = get_n_classes(hparams)

    if test_mode:
        hparams['test_mode'] = True

    print('\ncreating model...')
    net = get_model(hparams, n_classes, saved_model_path=saved_model_path, strategy=None)

    print('\nloading weights...')
    if epoch_to_load == 0:
        weights_filename = 'model_weights_init.h5'
    else:
        weights_filename = f'ckpt_ep{epoch_to_load:03d}.h5'

    net.load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename))

    if test_mode:
        print(f"test_mode={hparams['test_mode']}, setting trainable=False for all layers")
        for layer in net.layers:
            layer.trainable = False
    else:
        print(f"test_mode={hparams['test_mode']}, all layers are trainable")

    if print_summary:
        net.summary()

    metric_dict = {}
    for layer in net.output_names:
        metric_dict[layer] = [tf.keras.metrics.categorical_accuracy,
                              tf.keras.metrics.top_k_categorical_accuracy]
    
    net.compile(metrics=metric_dict)

    return net


def localdir_modulespec(module_name, dir_path):
    '''This function allows us to import functions from remote folders (e.g., needed when loading the model
    from root/save_dir/.../_code_used_for_training) - used in root/task_helper_fuctions.load_model_from_path'''
    file_path = dir_path + f'/{module_name}.py'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_n_classes(hparams=None, dataset_path=None, dataset_subset=None):

    if 'simclr' in hparams['model_name'].lower() and not 'finetune' in hparams['model_name'].lower():
        return 0

    if [hparams, dataset_path] == [None, None]:
        raise Exception('hparams or dataset_path must be passed as arguments in get n_classes')

    dataset_path = hparams['dataset'] if dataset_path is None else dataset_path
    with h5py.File(dataset_path, "r") as f:
        print(f'getting n_classes from {dataset_path}')
        if hparams['embedding_target']:
            print(f'\tusing embeddings dimension from {hparams["target_dataset_name"]}')
            return f['train'][hparams['target_dataset_name']][0].shape[-1]
        else:
            print(f'\tusing len(categories)')
            return f['categories'][:].shape[0]


def custom_load_checkpoint(hparams, net, model_savedir, epoch_to_load):

    ckpt_path = os.path.join(model_savedir, 'training_checkpoints')

    if epoch_to_load > 0:
        # in this case, the user manually picks which epoch to load, with the option of adding new readout units (e.g.,
        # to add new categories to the existing ones), or reinitializing the readout (for finetuning).
        if 'simclr' in hparams['model_name'] and 'finetune' not in hparams['model_name']:
            # Load encoder weights
            encoder_ckpt_path = os.path.join(ckpt_path, f'ckpt_ep{hparams["start_epoch"]:03d}.h5')
            net.encoder.load_weights(encoder_ckpt_path)
            # Load projector weights, Assuming net.projectors is a list of projector models: actually only 2: 0&1
            for i, projector in enumerate(net.projectors):
                projector_ckpt_path = os.path.join(ckpt_path, f'projector_ckpt_ep{hparams["start_epoch"]:03d}_{i}.h5')
                projector.load_weights(projector_ckpt_path)
        else:
            weights_to_load = os.path.join(ckpt_path, 'ckpt_ep{:03d}.h5'.format(epoch_to_load))
            print(f'loading weight from {weights_to_load}')
            net.load_weights(weights_to_load)
    else:
        raise Exception(f'Trying to load a checkpoint for epoch {epoch_to_load}, which is impossible.')
    

def save_init_model(hparams, net, model_savedir):
    init_ckpt_path = os.path.join(model_savedir, 'training_checkpoints', 'model_weights_init.h5')
    init_model_path = os.path.join(model_savedir, 'saved_model', f"{hparams['model_name']}_init")
    print(f'Saving initial checkpoint to {init_ckpt_path} and init full model to {init_model_path}')
    if 'simclr' in hparams['model_name'] and 'finetune' not in hparams['model_name']:
        # Save encoder weights
        net.encoder.save_weights(init_ckpt_path[:-3] + f"_encoder.h5")
        # Save projector weights, Assuming net.projectors is a list of projector models: actually only 2: 0&1
        for i, projector in enumerate(net.projectors):
            projector_ckpt_path = init_ckpt_path[:-3] + f"_projector_{i}.h5"
            projector.save_weights(projector_ckpt_path)
    else:
        net.save_weights(init_ckpt_path)

    try:
        save_full_model(0, hparams, net, init_model_path)
    except:
        print('Cannot save full model. This is probably because it contains custom layers, which need a bit of work'
                ' to be saved in this format. The checkpoint format should still work.')


class BatchLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, n_batches, hparams):

        print(f'Creating learning rate schedule, in {hparams["learning_rate_schedule"]} mode')
        self.i_epoch = hparams['start_epoch']

        if hparams['total_epochs'] > 0:
            total_epochs = hparams['total_epochs']
        else:
            total_epochs = hparams['n_epochs']
        assert(total_epochs > hparams['n_warmup_epochs'])

        self.n_batches = n_batches

        total_steps = np.arange(n_batches * (total_epochs-hparams['n_warmup_epochs']), dtype=np.float32)
        total_steps /= total_steps.max()
        if hparams['n_warmup_epochs'] > 0:
            self.schedule = tf.linspace(1/(n_batches * hparams['n_warmup_epochs']), 1, n_batches * hparams['n_warmup_epochs'])
        if hparams['learning_rate_schedule'] == 'cosine':
            cosine_decayed = 0.5 * (1.0 + math_ops.cos(constant_op.constant(math.pi) * total_steps))
            if hparams['n_warmup_epochs'] == 0:
                self.schedule = cosine_decayed
            else:
                self.schedule = k.concatenate((self.schedule, cosine_decayed))
        elif hparams['learning_rate_schedule'] == 'cosine_restarts':
            n_cycles = 10
            cosine_restarts = 0.5 * (1.0 + math_ops.cos(constant_op.constant(n_cycles*math.pi*(tf.sqrt(total_steps)%(1/n_cycles)))))
            if hparams['n_warmup_epochs'] == 0:
                self.schedule = cosine_decayed
            else:
                self.schedule = k.concatenate((self.schedule, cosine_restarts))
        elif hparams['learning_rate_schedule'] == 'none':
            if hparams['n_warmup_epochs'] == 0:
                self.schedule = tf.ones(len(total_steps))
            else:
                self.schedule = k.concatenate((self.schedule, tf.ones(len(total_steps))))
        self.schedule = k.reshape(self.schedule, (total_epochs, n_batches)) * hparams['learning_rate']

    def on_batch_begin(self, batch, logs={}):
        current_lr = self.schedule[self.i_epoch, min(self.n_batches - 1, batch)]
        k.set_value(self.model.optimizer.lr, current_lr)
        tf.summary.scalar('lr', data=current_lr, step=batch)

    def on_epoch_end(self, epoch, logs={}):
        self.i_epoch += 1
        

class TensorBoardFix(tf.keras.callbacks.TensorBoard):
    """
    This fixes incorrect step values when using the TensorBoard callback
    """

    def on_train_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)

    def on_test_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_test_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._val_step)


# Define a directory to save the encoder weights
class SimCLRModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, encoder_save_dir, save_freq=10, start_epoch=0, projectors_save_dir=None):
        super(SimCLRModelCheckpoint, self).__init__()
        self.encoder_save_dir = encoder_save_dir
        self.projectors_save_dir = projectors_save_dir
        self.save_freq = save_freq
        self.epoch_count = start_epoch

        # Ensure save directories exist
        os.makedirs(self.encoder_save_dir, exist_ok=True)
        if self.projectors_save_dir is not None:
            os.makedirs(self.projectors_save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        if self.epoch_count % self.save_freq == 0:
            # Save encoder weights
            encoder_path = os.path.join(self.encoder_save_dir, f"ckpt_ep{self.epoch_count:03d}.h5")
            self.model.encoder.save_weights(encoder_path)

            if self.projectors_save_dir is not None:
                # Save projectors weights
                projectors_path = os.path.join(self.projectors_save_dir, f"projector_ckpt_ep{self.epoch_count:03d}.h5")
                self.model.projectors[0].save_weights(projectors_path[:-3]+"_0.h5")
                self.model.projectors[1].save_weights(projectors_path[:-3]+"_1.h5")


# The functions below are used for testing on images that are not stored in .h5 format.
# During training and in most testing cases, images are in our standard .h5 format, and
# processing is done automatically. But for the orientation selectivity analysis, we
# create grating stimuli that are not in this standard .h5 formal. These functions can
# be used to make sure these non-h5 images are preprocessed correctly.

def preprocess_batch(batch, hparams):
    # during training, preprocessing is done in the HDF5generator. we to do it here, too.
    # img_normalization: str or None, '[-1,1]' normalizes to [-1,1], 'z_scoring' normalizes to mean=0, std=1
    for i, img in enumerate(batch):
        batch[i] = tf_preprocess_image(img, hparams=hparams)
    batch = tf.cast(batch, tf.float32)  # to avoid tf warning
    return batch


def tf_preprocess_image(image, hparams):
    '''Convert to float32, and scale to 0,1'''
    image = tf.cast(image, tf.float32)  # to avoid tf warning
    image = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/255.)(image)  # REMOVE IF if image is already [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf_normalize(image, hparams['image_normalization'])
    return image


def tf_normalize(image, img_normalization):
    '''normalize an image
    img_normalization: str or None, '[-1,1]' normalizes to [-1,1], 'z_scoring' normalizes to mean=0, std=1'''

    if img_normalization == 'z_scoring':
        image = tf.image.per_image_standardization(image)
    elif img_normalization == '[-1,1]':
        image = tf.keras.layers.experimental.preprocessing.Rescaling(scale=2, offset=-1)(image)  # assumes we start from a [0,1] image
    elif img_normalization is None:
        pass
    else:
        raise Exception("Please use '[-1,1]' or 'z_scoring' for the normalization argument")
    return image