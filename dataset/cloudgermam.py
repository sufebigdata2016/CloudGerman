import os
import tensorflow as tf
import h5py
import glob

SPLITS_TO_SIZES = {'train': 352366, 'validation': 24119}

_NUM_CLASSES = 17

_SPLIT_NAME_DICT = {'train': 'training.h5', 'validation': 'validation.h5'}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

_FILE_PATTERN = 'CloudGerman_%s_*.tfrecord'

_SEN1_SHAPE = (32,32,8)
_SEN2_SHAPE = (32,32,10)

FLAGS = tf.app.flags.FLAGS

class generator:
    def __init__(self, file, what):
        self.file = file
        self.what = what

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for i in hf[self.what]:
                yield i


def get_split(num_epochs, batch_size, dataset_dir, split_name):
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
    dataset1 = tf.data.Dataset.from_generator(
        generator(os.path.join(dataset_dir, _SPLIT_NAME_DICT[split_name]), 'sen1'),
        'float32'
    ).batch(batch_size=batch_size)
    dataset2 = tf.data.Dataset.from_generator(
        generator(os.path.join(dataset_dir, _SPLIT_NAME_DICT[split_name]), 'sen2'),
        'float32'
    ).batch(batch_size=batch_size)
    dataset3 = tf.data.Dataset.from_generator(
        generator(os.path.join(dataset_dir, _SPLIT_NAME_DICT[split_name]), 'label'),
        'float32'
    ).batch(batch_size=batch_size)
    dataset = tf.data.Dataset.zip((dataset1, dataset2, dataset3))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    iterator.num_classes = _NUM_CLASSES
    iterator.num_samples = SPLITS_TO_SIZES[split_name]
    return iterator


def _parse_function(example_proto):
    features = {"sen1": tf.FixedLenFeature(_SEN1_SHAPE, tf.float32),
                "sen2": tf.FixedLenFeature(_SEN2_SHAPE, tf.float32),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["sen1"], parsed_features["sen2"], parsed_features["label"]


def get_split1(num_epochs, batch_size, dataset_dir, split_name, num_readers=None, file_pattern=None):
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    filenames = glob.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_readers)
    dataset = dataset.map(_parse_function, num_parallel_calls=num_readers)  # Parse the record into tensors.
    dataset = dataset.repeat(num_epochs)  # Repeat the input indefinitely.
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    # sess.run(iterator.initializer)
    iterator.num_classes = _NUM_CLASSES
    iterator.num_samples = SPLITS_TO_SIZES[split_name]
    return iterator
