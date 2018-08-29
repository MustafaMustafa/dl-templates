#### Project code structure
- `./models`: contains models classes  
- `./hparams`: contains yparams wrapper and hyperparameter configuration files  
- `./data`: contains data preparation and pipeline class  

  
#### Training:
```bash
python train.py hparams/cnn.yaml baseline
```
  
See the YAML configuration file for how to setup an experiment parameters.  
