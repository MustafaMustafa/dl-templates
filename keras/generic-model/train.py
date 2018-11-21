import os
import sys
import logging

import logging_utils
logging_utils.config_logger()
from tensorflow import keras
from models import my_model
from data.data_pipeline import get_datasets
from hparams.yparams import YParams

def train(params, callbacks):
    """ Training loop """

    model = my_model.model(params)
    optimizer = my_model.optimizer(params.optimizer)
    checkpoint, filepath = my_model.checkpoint(params.experiment_dir)
    callbacks.append(checkpoint)

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()

    train_dataset, valid_dataset = get_datasets(params)
    model.fit(train_dataset,
              steps_per_epoch=params.train_data_size//params.batchsize,
              epochs=params.epochs,
              validation_data=valid_dataset,
              validation_steps=params.valid_data_size//params.batchsize,
              callbacks=callbacks)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        logging.error("Usage", sys.argv[0], "configuration_YAML_file", "configuration")
        exit()

    # load hyperparameters
    params = YParams(os.path.abspath(sys.argv[1]), sys.argv[2])

    os.mkdir(params.experiment_dir)
    log_filename=params.experiment_dir+'/output.log'
    logging_utils.log_to_file(logger_name=None, log_filename=log_filename)
    logging_utils.log_versions()
    params.log()

    keras_logger = keras.callbacks.CSVLogger(log_filename, separator=',', append=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir=params.experiment_dir)

    train(params, [keras_logger, tensorboard])
