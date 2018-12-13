# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from dataset import cloudgermam
import time
from datetime import datetime
import os
import shutil

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', './Log_try/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy. Note For '
                            'historical reasons loss from all clones averaged '
                            'out and learning rate decay happen per clone '
                            'epochs')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 5,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

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

tf.app.flags.DEFINE_integer(
    'quantize_delay', -1,
    'Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')

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
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.999,
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
    'dataset_dir', './Data', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v2_50', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 72, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 32, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 5000000,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################
tf.app.flags.DEFINE_integer(
    'num_epochs', 5000,
    'Training epochs')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/duoduo/projects/CloudGerman/Log_try',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'resnet_v2_50/logits',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    Raises:
      ValueError: if
    """
    # Note: when num_clones is > 1, this will actually have each clone to go
    # over each epoch FLAGS.num_epochs_per_decay times. This is different
    # behavior from sync replicas and is expected to produce different results.
    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                      FLAGS.batch_size)

    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
    return optimizer


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.
    Returns:
      An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def delete_old_model(checkpoint_path):
    del_list = sorted(tf.gfile.Glob(os.path.join('/'.join(checkpoint_path.split('/')[:-1]), '*.ckpt*')))
    if len(del_list) > 15:
        for f in del_list[:3]:
            tf.gfile.Remove(f)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        # get dataset
        with tf.device('/cpu:0'):
            dataset_train = cloudgermam.get_split1(FLAGS.dataset_dir, 'train',
                                                   FLAGS.batch_size, num_epochs=FLAGS.num_epochs,
                                                   num_readers=FLAGS.num_readers)
            dataset_validation = cloudgermam.get_split1(FLAGS.dataset_dir, 'validation',
                                                        FLAGS.batch_size, num_epochs=1,
                                                        num_readers=FLAGS.num_readers)
        # global_step = tf.train.create_global_step()
        # Calculate the learning rate schedule.
        # num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
        #                          FLAGS.batch_size / FLAGS.num_clones)
        # decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
        learning_rate = _configure_learning_rate(dataset_train.num_samples, global_step)
        optimizer = _configure_optimizer(learning_rate)

        tower_grads = []
        is_training = tf.placeholder(tf.bool, shape=())
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_clones):
                # with tf.device('/gpu:%d' % i):
                # with tf.name_scope('gpu_%d' % i) as scope:
                ######################
                # Select the network #
                ######################
                network_fn = nets_factory.get_network_fn(
                    FLAGS.model_name,
                    num_classes=(dataset_train.num_classes - FLAGS.labels_offset),
                    weight_decay=FLAGS.weight_decay,
                    is_training=True)
                # loss = tower_loss(scope, image_batch, label_batch)
                # Reuse variables for the next tower.
                # with tf.device('/cpu:0'):
                ##############################################################
                # Create a dataset provider that loads data from the dataset #
                ##############################################################
                sen1, sen2, labels = tf.cond(is_training, lambda: dataset_train.get_next(),
                                             lambda: dataset_validation.get_next())
                # train_image_size = FLAGS.train_image_size or network_fn.default_image_size
                # sen1.set_shape([FLAGS.batch_size, 32, 32, 8])
                # sen2.set_shape([FLAGS.batch_size, 32, 32, 10])
                # images = tf.concat((sen1, sen2), axis=3)
                images = sen2[:, :, :, :3]
                # images = image_preprocessing_fn(images, train_image_size, train_image_size)
                labels = tf.one_hot(
                    labels, dataset_train.num_classes - FLAGS.labels_offset)
                # labels.set_shape([FLAGS.batch_size, dataset.num_classes - FLAGS.labels_offset])

                logits, end_points = network_fn(images)
                if 'AuxLogits' in end_points:
                    Auxloss = tf.losses.softmax_cross_entropy(
                        labels, end_points['AuxLogits'],
                        label_smoothing=FLAGS.label_smoothing, weights=0.4,
                        scope='aux_loss')
                loss = tf.losses.softmax_cross_entropy(
                    labels, logits, label_smoothing=FLAGS.label_smoothing, weights=1.0)
                loss_val = tf.losses.softmax_cross_entropy(
                    labels, logits, label_smoothing=FLAGS.label_smoothing, weights=1.0)
                # accuracy = tf.metrics.accuracy(tf.argmax(labels,1), tf.argmax(logits,1))
                # accuracy_val = tf.metrics.accuracy(tf.argmax(labels,1), tf.argmax(logits,1))

                tf.get_variable_scope().reuse_variables()

                # Calculate the gradients for the batch of data on this CIFAR tower.
                grads = optimizer.compute_gradients(loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)

        # with tf.device('/gpu:0'):
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        # grads = tower_grads[0]
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
        # # print(tf.Session().run(tf.trainable_variables()))
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Retain the summaries from the final tower.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)  # , scope)
        # Add a summary to track the learning rate.

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))
        # summaries.append(tf.summary.scalar('accuracy', accuracy))
        summaries.append(tf.summary.scalar('loss', loss))

        # Build the summary operation from the last tower summaries.
        summary_op_train = tf.summary.merge(summaries)
        summary_op_validation = tf.summary.merge([tf.summary.scalar('loss_val', loss_val),
                                                  # tf.summary.scalar('accuracy_val', accuracy_val)
                                                  ])

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path, latest_filename=None)
        if latest_checkpoint_path:
            print('restoring from %s' % latest_checkpoint_path)
            saver.restore(sess, latest_checkpoint_path)
            print('restored')
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        start_step = 0 if not latest_checkpoint_path else int(latest_checkpoint_path.split('-')[-1])
        for step in xrange(start_step, FLAGS.max_number_of_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss], feed_dict={is_training: True})
            duration = time.time() - start_time
            # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_clones
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_clones
                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 50 == 0:
                summary_str_train = sess.run(summary_op_train, feed_dict={is_training: True})
                try:
                    summary_str_validation = sess.run(summary_op_validation, feed_dict={is_training: False})
                except:
                    sess.run(dataset_validation.initializer)
                    summary_str_validation = sess.run(summary_op_validation, feed_dict={is_training: False})
                summary_writer.add_summary(summary_str_train, step)
                summary_writer.add_summary(summary_str_validation, step)

            # Save the model checkpoint periodically.
            if step % 10000 == 0 or (step + 1) == FLAGS.max_number_of_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                delete_old_model(checkpoint_path)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
