# Created by Hui Guan Dec.21, 2017. 
# This script report the number of parameters for a network given a configuration 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import mpi module, must be first import
from mpi4py import MPI


import tensorflow as tf
from tensorflow.python.client import timeline

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import re, time,  math , os, sys
from datetime import datetime
from pprint import pprint

from train_helper import *

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'summary_every_n_steps', 50,
    'The frequency with which summary op are done.')

tf.app.flags.DEFINE_integer(
    'evaluate_every_n_steps', 100,
    'The frequency with which evaluation are done.')

tf.app.flags.DEFINE_integer(
    'runmeta_every_n_steps', 1000,
    'The frequency with which RunMetadata are done.')



tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')



######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')


tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

# added by Hui Guan
tf.app.flags.DEFINE_string(
    'net_name_scope_checkpoint', 'resnet_v1_50',
    'The name scope for the saved previous trained network')

tf.app.flags.DEFINE_string(
    'net_name_scope_pruned', 'resnet_v1_50_pruned',
    'The name scope for the pruned network in the current graph')

tf.app.flags.DEFINE_string(
    'kept_percentages', '0.5',
    'The numbers of filters to keep')

tf.app.flags.DEFINE_integer(
    'configuration_index', 0,
    'The index of configurations to evaluate in the main function')

tf.app.flags.DEFINE_integer(
    'num_configurations', 100,
    'The number of configurations to evaluate')
tf.app.flags.DEFINE_integer(
    'total_num_configurations', 100,
    'The total number of configurations to evaluate')

tf.app.flags.DEFINE_integer(
    'start_configuration_index', 0,
    'The start index of configurations to evaluate in the main function')

tf.app.flags.DEFINE_string(
    'configuration_type', 'special',
    'The way to generate some configurations to evaluate. One of "special", "sample", "rank"')

tf.app.flags.DEFINE_integer(
    'test_batch_size', 32, 'The number of samples in each batch for test dataset.')

tf.app.flags.DEFINE_string(
    'train_dataset_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'test_dataset_name', 'val', 'The name of the train/test split.')

FLAGS = tf.app.flags.FLAGS

def main(_):
    tic = time.time() 
    tf.logging.set_verbosity(tf.logging.INFO)

    # init 
    net_name_scope_pruned = FLAGS.net_name_scope_pruned
    net_name_scope_checkpoint = FLAGS.net_name_scope_checkpoint
    indexed_prune_scopes_for_units = valid_indexed_prune_scopes_for_units
    kept_percentages = sorted(map(float, FLAGS.kept_percentages.split(',')))

    num_options = len(kept_percentages)
    num_units = len(indexed_prune_scopes_for_units)
    print('num_options=%d, num_blocks=%d' %(num_options, num_units))
    print('HG: total number of configurations=%d' %(num_options**num_units))

    # find the  configurations to evaluate 
    if FLAGS.configuration_type =='sample':
        configs = get_sampled_configurations(num_units, num_options, FLAGS.total_num_configurations)
    elif FLAGS.configuration_type == 'special':
        configs = get_special_configurations(num_units, num_options)
    num_configurations = len(configs)
    
    #Getting MPI rank integer
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank >= num_configurations:
	print("ERROR: rank(%d) > num_configurations(%d)" %(rank, num_configurations))
        return  
    FLAGS.configuration_index = FLAGS.start_configuration_index + rank
    config = configs[FLAGS.configuration_index]
    if FLAGS.configuration_index >= num_configurations:
        print('configuration_index >= num_configurations', FLAGS.configuration_index, num_configurations)
        return 
    print('HG: kept_percentages=%s, start_config_index=%d, num_configs=%d, rank=%d, config_index=%d' \
           %(str(kept_percentages), FLAGS.start_configuration_index, num_configurations, rank, FLAGS.configuration_index))

    # prepare for evaluate the number of parameters with the specific config 
    combination = config 
    indexed_prune_scopes, kept_percentage = config_to_indexed_prune_scopes(combination, indexed_prune_scopes_for_units, kept_percentages)
    prune_scopes = indexed_prune_scopes_to_prune_scopes(indexed_prune_scopes, net_name_scope_checkpoint)
    shorten_scopes = indexed_prune_scopes_to_shorten_scopes(indexed_prune_scopes, net_name_scope_checkpoint)
    reinit_scopes = [re.sub(net_name_scope_checkpoint, net_name_scope_pruned, v) for v in prune_scopes+shorten_scopes]

    # prepare file system 
    eval_dir = os.path.join(FLAGS.train_dir, "id"+str(FLAGS.configuration_index))
    prepare_file_system(eval_dir)

    # functions to write logs
    # def write_log_info(info):
    #     with open(os.path.join(FLAGS.train_dir, 'log.txt'), 'a') as f:
    #             f.write(info+'\n')
    def write_detailed_info(info):
        with open(os.path.join(eval_dir, 'eval_details.txt'), 'a') as f:
            f.write(info+'\n')

    info = 'eval_dir:'+eval_dir+'\n'
    info += 'options:'+FLAGS.kept_percentages+'\n'
    info += 'combination: '+ str(combination)+'\n'
    info += 'indexed_prune_scopes: ' + str(indexed_prune_scopes)+'\n'
    info += 'kept_percentage: ' + str(kept_percentage)
    print(info)
    write_detailed_info(info)

    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        ######################
        # Select the dataset #
        ######################
        test_dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.test_dataset_name, FLAGS.dataset_dir)
        test_images, test_labels = test_inputs(test_dataset, deploy_config, FLAGS)

        ######################
        # Select the network#
        ######################
        network_fn_pruned = nets_factory.get_network_fn_pruned(
                FLAGS.model_name,
                num_classes=(test_dataset.num_classes - FLAGS.labels_offset),
                weight_decay=FLAGS.weight_decay)

        ####################
        # Define the model #
        ####################
        prune_info = indexed_prune_scopes_to_prune_info(indexed_prune_scopes, kept_percentage)
        print('HG: prune_info:')
        pprint(prune_info)

        logits, _ = network_fn_pruned(test_images, 
                                      prune_info = prune_info, 
                                      is_training=False, 
                                      is_local_train=False, 
                                      reuse_variables=False,
                                      scope = net_name_scope_pruned)

        correct_prediction = add_correct_prediction(logits, test_labels)
        model_variables = get_model_variables_within_scopes()
        name_size_strs = [x.op.name+'\t'+str(x.get_shape().as_list()) for x in model_variables]
        write_detailed_info('\n'.join(name_size_strs))


    total_time = time.time()-tic 
    info = 'Evaluate network total_time(s)=%.3f \n' %(total_time)
    print(info)
    write_detailed_info(info)

    # evaluate(config)

if __name__ == '__main__':
    tf.app.run()










