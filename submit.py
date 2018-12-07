# coding=utf8
import tensorflow as tf
from dataset import cloudgermam
from nets import nets_factory
from deployment import model_deploy
from preprocessing import preprocessing_factory


tf.app.flags.DEFINE_string(
    'checkpoint_path', './Log2/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'round1_test', 'The name of the train/validation split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './Data', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v2_50', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'submit_dir', './submit/submit2.csv', 'submit dir')

tf.app.flags.DEFINE_integer(
    'num_readers', 12,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'test_image_size', 32, 'Test image size')

FLAGS = tf.app.flags.FLAGS





def main(_):
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=1,
        clone_on_cpu=False,
        replica_id=0,
        num_replicas=1,
        num_ps_tasks=0)

    with tf.device(deploy_config.variables_device()):
        dataset = cloudgermam.get_split1(FLAGS.dataset_dir, FLAGS.dataset_split_name,
                                         num_readers=FLAGS.num_readers, file_pattern='%s_*.tfrecord')

    network_fn = nets_factory.get_network_fn(FLAGS.model_name,
                                             num_classes=17,
                                             is_training=False)

    sen2 = dataset.get_next()[1]
    # image = tf.concat((sen1, sen2), axis=3)
    images = sen2[:, :, :, :3]
    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    test_image_size = FLAGS.test_image_size or network_fn.default_image_size
    images = image_preprocessing_fn(images, test_image_size, test_image_size)

    logits, _ = network_fn(images)
    logits = tf.argmax(logits, 1)
    logits = tf.one_hot(logits, 17)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))
        with open(FLAGS.submit_dir, 'a+') as f:
            while True:
                try:
                    logits_ = sess.run(logits)
                    f.write(','.join(list(map(lambda x:str(int(x)), logits_.ravel()))) + '\n')
                except:
                    break


if __name__ == '__main__':
    tf.app.run()




