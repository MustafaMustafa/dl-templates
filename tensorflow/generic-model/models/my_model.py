__author__ = "Mustafa Mustafa"
__email__  = "mmustafa@lbl.gov"

import tensorflow as tf

# pylint: disable-msg=C0103
# pylint: disable=too-many-instance-attributes

class MyModel(object):
    """ My Model"""

    def __init__(self, params, input_x=None, is_training=True):

        self._params = params
        self._is_training = is_training
        self._data_format = params.data_format

        if input_x is not None:
            self.input_x = input_x
        else:
            self.input_x = tf.placeholder(dtype=tf.float32, name='x', shape=params.input_shape)

        with tf.variable_scope('training_counters', reuse=tf.AUTO_REUSE) as _:
            self.global_step = tf.train.get_or_create_global_step()

        self.build_graph()
        self.loss = None
        self.optimizer = None

    def build_graph(self):
        """ network """

        #remove the following two lines after you implement your graph
        print("MyModel.build_graph() is not implemented. No graph to work with!")
        raise NotImplementedError

        # build network graph here
        # usage of variables scopes is highly encouraged

        # set self.predictions to the network output
        self.predictions = None

    def define_loss(self, labels):
        """ define loss """

        with tf.name_scope('loss'):
            # example loss, change to yours
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=labels, predictions=self.predictions))

    def define_optimizer(self):
        """ build optimizer op """

        with tf.variable_scope('optimizer') as _:
            # example optimizer, change to yours
            self.optimizer = tf.train.AdamOptimizer(self._params.learning_rate).\
                                      minimize(self.loss, global_step=self.global_step)
