import os

### MODEL-SPECIFIC PARAMETERS ###

# # multihot model
# this_model = 'blt_vNet_half_channels'  # 'blt_vNet'
# model_name_suffix = '_multihot_Nov23'
# target_dataset_name = 'img_multi_hot'
# model_output_activation = 'sigmoid'
# batch_size = 256
# # unused (dummies)
# simclr_temperature = 0

# # mpnet model
# this_model = 'blt_vNet_half_channels'  # 'blt_vNet'
# model_name_suffix = '_mpnet_Nov23'
# target_dataset_name = 'all_mpnet_base_v2_mean_embeddings'
# model_output_activation = 'no_model_output_activation'
# batch_size = 256
# # unused (dummies)
# simclr_temperature = 0

# simclr model
this_model = 'simclr_blt_vNet_half_channels'
model_name_suffix = '_simclr_Nov23'
simclr_temperature = 0.1
batch_size = 96  # this can fit on a single 80GB GPU
# unused (dummies)
target_dataset_name = None
model_output_activation = None

### GENERAL PARAMETERS ###

norm_type = 'LN'
norm_axes = -1
dropout_rate = 0
recurrent_steps = 6
image_size = 128

dataset = '/share/klab/datasets/ms_coco_embeddings_square256_proper_chunks.h5'  # in this case images are square and we can load everything in RAM
embedding_loss = 'cosine'  # MSE or cosine
embedding_target = True
save_dir = './save_dir'

adam_epsilon = 1e-1
lr = 5e-2
n_warmup_epochs = 10
learning_rate_schedule = 'cosine'
regularize = 1e-6
start_epoch = 0
n_epochs = 400
use_mixed_precision = False
use_class_weights = False

task_script = 'task.py'

### RUN ###

run_string = f'python {task_script} ' +\
             f'--model-name {this_model} ' +\
             f'--model-name-suffix {model_name_suffix} ' +\
             f'--embedding-target {embedding_target} ' +\
             f'--embedding-loss {embedding_loss} ' +\
             f'--target-dataset-name {target_dataset_name} ' +\
             f'--simclr-temperature {simclr_temperature} ' +\
             f'--model-output-activation {model_output_activation} ' +\
             f'--n-recurrent-steps {recurrent_steps} ' +\
             f'--calculate-class-weights {use_class_weights} ' +\
             f'--norm-type {norm_type} ' +\
             f'--norm-axes {norm_axes} ' +\
             f'--dropout-rate {0} ' +\
             f'--dataset {dataset} ' +\
             f'--batch-size {batch_size} ' +\
             f'--image-size {image_size} ' +\
             f'--learning-rate {lr} ' +\
             f'--n-warmup-epochs {n_warmup_epochs} ' +\
             f'--learning-rate-schedule {learning_rate_schedule} ' +\
             f'--regularize {regularize} ' +\
             f'--optim-epsilon {adam_epsilon} ' +\
             f'--gpu-ids 0 ' +\
             f'--start-epoch {start_epoch} ' +\
             f'--n-epochs {n_epochs} ' +\
             f'--save-dir {save_dir} ' +\
             f'--numpy-seed 1 ' +\
             f'--tensorflow-seed 1 ' +\
             f'--use-mixed-precision {use_mixed_precision} '

os.system(run_string)
