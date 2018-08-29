import os
import glob
import numpy as np
import tensorflow as tf

def parse_tfevents(filepath):

    history = {}
    for event in tf.train.summary_iterator(filepath):
        for value in event.summary.value:

            if value.tag not in history:
                history[value.tag] = {'steps': [], 'values': []}

            if value.HasField('simple_value'):
                history[value.tag]['steps'].append(event.step)
                history[value.tag]['values'].append(value.simple_value)

    return history

def get_training_history(model_dir):
    train_hist = parse_tfevents(glob.glob(os.path.join(model_dir, '*.tfevents.*'))[0])
    valid_hist = parse_tfevents(glob.glob(os.path.join(model_dir, 'eval/*.tfevents.*'))[0])

    return train_hist, valid_hist
