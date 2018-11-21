""" Yaml file parser derivative of tensorflow.contrib.training.HParams """
""" Original code: https://hanxiao.github.io/2017/12/21/Use-HParams-and-YAML-to-Better-Manage-Hyperparameters-in-Tensorflow/ """

from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
import logging

class YParams(HParams):
    """ Yaml file parser derivative of HParams """

    def __init__(self, yaml_fn, config_name, print_params=False):

        self._yaml_fn = yaml_fn
        self._config_name = config_name

        super(YParams, self).__init__()
        if print_params: print("------------------ HParams ------------------")
        with open(yaml_fn) as yamlfile:
            for key, val in YAML().load(yamlfile)[config_name].items():
                if print_params: print(key, val)
                if val =='None':
                    val = None
                self.add_hparam(key, val)
        if print_params: print("---------------------------------------------")

    def log(self):
        logging.info("------------------ HParams ------------------")
        logging.info("Configuration file: "+str(self._yaml_fn))
        logging.info("Configuration name: "+str(self._config_name))
        with open(self._yaml_fn) as yamlfile:
            for key, val in YAML().load(yamlfile)[self._config_name].items():
                logging.info(str(key) + ' ' + str(val))
        logging.info("---------------------------------------------")
