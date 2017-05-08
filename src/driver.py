"""
Setup and execute sacred experiments that are saved in a specified mongodb tables. Model updates are stored in:
- ```PROJ_ROOT/model_results/mongo```
- ```PROJ_ROOT/model_results/saved_models```
- ```PROJ_ROOT/model_results/tensorboard```
- ```PROJ_ROOT/model_results/csv_results```
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

PROJ_ROOT = os.path.join(os.pardir, os.pardir)

# add local python functions
sys.path.append(os.path.join(PROJ_ROOT, "src"))
sys.path.append(os.path.join(PROJ_ROOT, "src", "model"))

from callbacks import *
from models import *
from run_experiment import *
from trainer import *
from image_utilities import load_images_from_directory
import keras.backend as K

from run_experiment import ex
from sacred.observers import MongoObserver
experiment_name = 'architectures_and_optimizers'
mongo_observer = MongoObserver.create(db_name=experiment_name)
ex.observers.append(mongo_observer)

import keras.backend as K
for trainer in ['ResNet50_FeatureExtractor',
                'ResNet50_FineTune_1skip',
                'ResNet50_FineTune_2skip',
                'ResNet50_FullTrain']:

    for optimizer in ['adam',
                      'nesterov',
                      'nadam',
                      'sgd',
                      'rmsprop',
                      'adagrad',
                      'adadelta']:

        C = { #  'experiment_name' : experiment_name,
            'trainer': trainer,
             'loss' : 'categorical_crossentropy',
             'metric': 'val_loss',
             'result_mode' : 'min',
             'epochs' : 50,
             'optimizer' : optimizer,
             'PROJ_ROOT' : PROJ_ROOT,
             'steps_per_epoch' : 100,
             'validation_steps' : 10}

        name = '+'.join('{}={}'.format(key, value) for key, value in C.items())
        # run = ex.run(config_updates=C, options={'--name': name})
        run = ex.run(config_updates={'trainer':trainer}, options={'--name': name})
    #     run = ex.run(config_updates=C)
        K.clear_session()
