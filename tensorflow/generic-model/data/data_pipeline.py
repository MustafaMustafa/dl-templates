__author__ = "Mustafa Mustafa"
__email__  = "mmustafa@lbl.gov"

import tensorflow as tf
import numpy as np

def get_tfrecords_dataset_input_fn(filenames, batchsize, epochs, variable_scope,
                                   shuffle_buffer_size=128, augment=True):
    """ creates a tf.data.Dataset and feeds and augments data from tfrecords

    Returns:
        data input function input_fn
        """
        
    def parse_fn(example):
        "Parse TFExample records and perform data augmentation"
        example_fmt = {'height': tf.FixedLenFeature([], tf.int64),
                       'width': tf.FixedLenFeature([], tf.int64),
                       'depth': tf.FixedLenFeature([], tf.int64),
                       'label': tf.FixedLenFeature([], tf.int64),
                       'image_raw': tf.FixedLenFeature([], tf.string, "")}
        parsed_record = tf.parse_single_example(example, example_fmt)

        # change to your data shapes!
        height = tf.cast(parsed_record['height'], tf.int32)
        width = tf.cast(parsed_record['width'], tf.int32)
        depth = tf.cast(parsed_record['depth'], tf.int32)

        X = tf.reshape(tf.decode_raw(parsed_record['image_raw'], tf.float32), [height, width, depth])
        label = tf.cast(parsed_record['label'], tf.int32)

        print(X.get_shape().as_list())

        # generic augmentation for images
        if augment is True:

            print("Augmenting data ...")
            # rotation
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k:
                X = tf.image.rot90(X, k=rot_k)

            # flipping
            if np.random.choice([0, 1]):
                X = tf.image.flip_left_right(X)

            if np.random.choice([0, 1]):
                X = tf.image.flip_up_down(X)

            if np.random.choice([0, 1]):
                X = tf.image.transpose_image(X)
        else:
            print("Not augmenting data...")
        
        return X, label

    def input_fn():
        """ create input_fn for Estimator training

        Returns:
            tf.Tensors of features and labels
        """

        with tf.variable_scope(variable_scope) as _:
            dataset = tf.data.TFRecordDataset(filenames,
                                              compression_type=None,
                                              buffer_size=2*shuffle_buffer_size,
                                              num_parallel_reads=32)
            dataset = dataset.shuffle(shuffle_buffer_size)
            dataset = dataset.repeat(epochs)
            dataset = dataset.map(map_func=parse_fn, num_parallel_calls=64)
            dataset = dataset.prefetch(256)
            dataset = dataset.batch(batchsize)

            data_it = dataset.make_one_shot_iterator()
            X, y = data_it.get_next()

        return X, y

    return input_fn
