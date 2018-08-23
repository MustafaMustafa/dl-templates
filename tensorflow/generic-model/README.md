#### Project code structure
- `./models`: contains models classes  
- `./hparams`: contains yparams wrapper and hyperparameter configuration files  
- `./logs`: this directory is created upon running an experiment, it contains model checkpoints  

  
#### Training:
```bash
python train.py hparams/my_model_configs.yaml baseline_experiment
```
  
See the YAML configuration file for how to setup an experiment parameters.  

#### Prediction:
```bash
python predict.py hparams/my_model_configs.yaml baseline_experiment
```
  
