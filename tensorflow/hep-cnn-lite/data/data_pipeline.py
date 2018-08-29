__author__ = "Mustafa Mustafa"
__email__  = "mmustafa@lbl.gov"

import tensorflow as tf
import h5py
import numpy as np
from data.iterator_initializer_hook import DatasetIteratorInitializerHook

def get_input_fn(filename, batchsize, epochs, variable_scope,
                 shuffle_buffer_size=128, augment=True):
    """ creates a tf.data.Dataset and feeds and augments data from an h5 file

    Returns:
        data input function input_fn
        """

    with h5py.File(filename) as _f:
        data_group = _f['all_events']
        features = np.expand_dims(data_group['hist'][:], axis=-1).astype(np.float32)
        labels = np.expand_dims(data_group['y'][:], axis=-1).astype(np.float32)
        _f.close()

    iterator_initializer_hook = DatasetIteratorInitializerHook()

    def input_fn():
        """ create input_fn for Estimator training

        Returns:
            tf.Tensors of features and labels
        """

        with tf.variable_scope(variable_scope) as _:
            features_placeholder = tf.placeholder(tf.float32, features.shape)
            labels_placeholder = tf.placeholder(tf.float32, labels.shape)

            dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,
                                                          labels_placeholder))
            dataset = dataset.shuffle(shuffle_buffer_size)
            dataset = dataset.repeat(epochs)
            dataset = dataset.prefetch(1)
            dataset = dataset.batch(batchsize)

            data_it = dataset.make_initializable_iterator()
            iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(data_it.initializer,
                                          feed_dict={features_placeholder: features,
                                                     labels_placeholder: labels})
            X, y = data_it.get_next()

        return X, y

    return input_fn, iterator_initializer_hook
