from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

# pylint: disable-msg=C0103
# pylint: disable=too-many-instance-attributes

def model(params):

    channels_axis = 1 if params.data_format == 'channels_first' else 3
    inputs = Input(tuple(params.input_shape))

    # compress frequency domain
    #outputs = Conv2D( ... )(inputs)
    #outputs = Conv2D( ... )(output)

    model = Model(inputs, outputs)

    return model

def optimizer(specs):
    if specs['name'].lower() == 'adam':
        return keras.optimizers.Adam(lr=specs['lr'])
    elif specs['name'].lower() == 'sgd':
        return keras.optimizers.SGD(lr=specs['lr'])

def checkpoint(experiment_dir):
    filepath = experiment_dir+'/model.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss')
    return checkpoint, filepath
