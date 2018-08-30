__author__ = "Mustafa Mustafa"
__email__  = "mmustafa@lbl.gov"

import os
import tensorflow as tf
from models.my_model import MyModel
from data.data_pipeline import get_tfrecords_dataset_input_fn
from hparams.yparams import YParams

def model_fn(features, labels, params, mode):
    """ Build graph and return EstimatorSpec """

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = MyModel(params, input_x=features, is_training=is_training)

    if mode is not tf.estimator.ModeKeys.PREDICT:
        # loss and optimizer are not needed for inference
        model.define_loss(labels)
        model.define_optimizer()
        model.define_train_op()

    return tf.estimator.EstimatorSpec(predictions=model.predictions,
                                      loss=model.loss,
                                      train_op=model.train_op,
                                      eval_metric_ops=None,
                                      mode=mode)
def main(argv):
    """ Training loop """

    if len(argv) != 3:
        print("Usage", argv[0], "configuration_YAML_file", "configuration")
        exit()

    # load hyperparameters
    params = YParams(os.path.abspath(argv[1]), argv[2])

    # build estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.experiment_dir,
                                       params=params)

    # create training data input pipeline
    train_input_fn = get_tfrecords_dataset_input_fn(params.train_data_files,
                                                    batchsize=params.batchsize,
                                                    epochs=params.epochs,
                                                    variable_scope='train_data_pipeline',
                                                    augment=params.augment_data)

    max_steps = (params.dataset_size//params.batchsize)*params.epochs
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=max_steps,
                                        hooks=[])

    # create validation data input pipeline
    valid_input_fn = get_tfrecords_dataset_input_fn(params.valid_data_files,
                                                    batchsize=params.batchsize,
                                                    epochs=params.epochs,
                                                    variable_scope='valid_data_pipeline',
                                                    augment=False)

    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, hooks=[])

    # train
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    tf.app.run()
