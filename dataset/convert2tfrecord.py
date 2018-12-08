#coding=utf8
import sys
import h5py

import numpy as np
import math
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', '../Data', '')

LABELS_FILENAME = 'labels.txt'
# The shards of output files
_NUM_SHARDS = 5

SEED = 7
# The height and width of each image.
_IMAGE_SIZE = 32

# The names of the classes.
_CLASS_NAMES = [
    'Compact_high_rise',
    'Compact_mid_rise',
    'Compact_low_rise',
    'Open_high_rise',
    'Open_mid_rise',
    'Open_low_rise',
    'Lightweight_low_rise',
    'Large_low_rise',
    'Sparsely_built',
    'Heavy_industry',
    'Dense trees',
    'Scattered_trees',
    'Bush_and_scrub',
    'Low_plants',
    'Bare_rock_or_paved',
    'Bare_soil_or_sand',
    'Water',
]

# shuffle
_NEW_ORDER = list(range(352366))
np.random.seed(SEED)
np.random.shuffle(_NEW_ORDER)
_SHUFFLE_MAP = {i:j for i,j in zip(list(range(352366)), _NEW_ORDER)}

S1_THRESHS = [(-25, 25), (-25, 25), (-70, 70), (-70, 70), (float("-inf"), 100), (float("-inf"), 1000),
              (-100, 100), (-100, 100)]  # 上界，下界
S2_THRESHS = [(float("-inf"), float("inf")), (float("-inf"), float("inf")), (float("-inf"), float("inf")),
              (float("-inf"), float("inf")), (float("-inf"), float("inf")), (float("-inf"), float("inf")),
              (float("-inf"), float("inf")), (float("-inf"), float("inf")), (float("-inf"), float("inf")),
              (float("-inf"), float("inf"))]
S1_MEANS_STDS = [(-3.1764201098502775e-05, 0.17253755233246393), (-7.0324133129218994e-06, 0.17233924335556297), (6.4947807422303307e-05, 0.45648411453353621), (2.9759795273244554e-05, 0.45325171128651598), (0.04255387426528303, 0.35996098865778275), (0.24984416768306306, 3.9778058737510902), (0.00076266614542442929, 0.42671998442466774), (0.00090506412854692508, 0.31763204200829365)]
S2_MEANS_STDS = [(0.1237569611768191, 0.03958795985905441), (0.1092774636368305, 0.047778262752410657), (0.10108552032678805, 0.066366167063719519), (0.11423986161140066, 0.063588749124974239), (0.15926566920230753, 0.07744387147984555), (0.18147236008771511, 0.091016350859215042), (0.1745740312291362, 0.092184665623869899), (0.19501607349634489, 0.1016458123394836), (0.15428468872076573, 0.099917730435192545), (0.10905050699570018, 0.087806325091228543)]


def preprocess(data, sen):
    import copy
    data = copy.deepcopy(data)
    if sen == "sen1":
        t = S1_THRESHS
        ms = S1_MEANS_STDS
    else:
        t = S2_THRESHS
        ms = S2_MEANS_STDS

    # 去异常
    lo_threshs = np.array([_[0] for _ in t])
    bool_matrix = data < lo_threshs.reshape(1, 1, -1)
    data = np.where(bool_matrix, np.tile(lo_threshs, (_IMAGE_SIZE, _IMAGE_SIZE, 1)), data)

    hi_threshs = np.array([_[1] for _ in t])
    bool_matrix = data > hi_threshs.reshape(1, 1, -1)
    data = np.where(bool_matrix, np.tile(hi_threshs, (_IMAGE_SIZE, _IMAGE_SIZE, 1)), data)

    # 标准化
    means = np.array([_[0] for _ in ms]).reshape(1, 1, -1)
    stds = np.array([_[1] for _ in ms]).reshape(1, 1, -1)
    data = (data - means) / stds

    return data


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_tfexample(sen1, sen2, image_size, split_name, class_id=None):
    if split_name == 'train' or split_name == 'validation':
        feature = {
            'sen1': float_feature(sen1),
            'sen2': float_feature(sen2),
            'label': int64_feature(class_id),
            'image_size': int64_feature(image_size)
        }
    else:
        feature = {
            'sen1': float_feature(sen1),
            'sen2': float_feature(sen2),
            'image_size': int64_feature(image_size)
        }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _add_to_tfrecord(filename, num_shards, tfrecord_filename, split_name, offset=0):
    data = h5py.File(filename, 'r')

    sen1 = data['sen1']
    sen2 = data['sen2']
    num_images = sen1.shape[0]
    num_per_shard = int(math.ceil(num_images / float(num_shards)))
    if split_name == 'train' or split_name == 'validation':
        labels = data['label']

    with tf.Graph().as_default():

        for shard_id in range(num_shards):
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, num_images)

            with tf.python_io.TFRecordWriter(
                    tfrecord_filename % ('0' * (4 - len(str(shard_id))) + str(shard_id))) as tfrecord_writer:
                for j in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                        filename, offset + j + 1, offset + num_images))  # \r：光标移到本行最左边
                    sys.stdout.flush()

                    if split_name == 'train':
                        j = _SHUFFLE_MAP[j]
                    if split_name == 'train' or split_name == 'validation':
                        example = to_tfexample(preprocess(sen1[j], "sen1").reshape(-1),
                                               preprocess(sen2[j], "sen2").reshape(-1),
                                               _IMAGE_SIZE,
                                               split_name,
                                               np.argmax(labels[j]))  # reshape(-1): 相当于ravel()
                    else:
                        example = to_tfexample(preprocess(sen1[j], "sen1").reshape(-1),
                                               preprocess(sen2[j], "sen2").reshape(-1),
                                               _IMAGE_SIZE,
                                               split_name)
                    tfrecord_writer.write(example.SerializeToString())

    return offset + num_images



def write_label_file(labels_to_class_names, dataset_dir, filename):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def main(_):
    if not tf.gfile.Exists(FLAGS.dataset_dir):
        tf.gfile.MakeDirs(FLAGS.dataset_dir)

    training_filename = '%s'.join(['%s/CloudGerman_%s_' % (FLAGS.dataset_dir, 'train'), '.tfrecord'])
    validation_filename = '%s'.join(['%s/CloudGerman_%s_' % (FLAGS.dataset_dir, 'validation'), '.tfrecord'])
    round1_test_filename = '%s'.join(['%s/round1_%s_' % (FLAGS.dataset_dir, 'test'), '.tfrecord'])
    source_training_filename = '%s/training.h5' % FLAGS.dataset_dir
    source_validation_filename = '%s/validation.h5' % FLAGS.dataset_dir
    source_round1_test_filename = '%s/round1_test_a_20181109.h5' % FLAGS.dataset_dir

    if tf.gfile.Exists(training_filename % '0000'):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    _add_to_tfrecord(source_training_filename,
                     num_shards=_NUM_SHARDS,
                     tfrecord_filename=training_filename,
                     split_name='train')

    # Next, process the validation data:
    _add_to_tfrecord(source_validation_filename,
                     num_shards=1,
                     tfrecord_filename=validation_filename,
                     split_name='validation')

    # Next, process the round1_test data:
    _add_to_tfrecord(source_round1_test_filename,
                     num_shards=1,
                     tfrecord_filename=round1_test_filename,
                     split_name='test')

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    write_label_file(labels_to_class_names, FLAGS.dataset_dir, LABELS_FILENAME)

    print('\nFinished converting the CloudGerman dataset!')


if __name__ == '__main__':
    tf.app.run()
