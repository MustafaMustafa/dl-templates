import tensorflow as tf

class PrintHook(tf.train.SessionRunHook):

    def __init__(self):
        super(PrintHook, self).__init__()

    def after_create_session(self, session, coord):
        for v in tf.trainable_variables():
            print(session.run(v))

    def end(self, session):
        for v in tf.trainable_variables():
            print(session.run(v))
