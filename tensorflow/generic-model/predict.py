import numpy as np

from hparams.yparams import YParams
from train import model_fn

def main(argv):
    """ prediction loop """

    if len(argv) != 3:
        print("Usage", argv[0], "configuration_YAML_file", "configuration")
        exit()

    # load hyperparameters
    params = YParams(os.path.abspath(argv[1]), argv[2])

    # build estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.experiment_dir,
                                       params=params)

    # load test data here:
    test_data = np.load('path/to/test/data.npy')
    test_data_input_fn = tf.estimator.inputs.numpy_input_fn(test_data, shuffle=False)

    for _prediction in estimator.predict(test_data_input_fn):
        # do something with prediction
        print(_prediction)

if __name__ == '__main__':
    tf.app.run()
