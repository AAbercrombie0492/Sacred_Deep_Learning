'''
Parent class that will be inherited by each specific trainer.
Takes care of data loading/processing, model compiling, and fitting.
'''

import numpy as np
import os
import sys
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

from image_utilities import load_images_from_directory

import dotenv

PROJ_ROOT = os.pardir
dotenv_path = os.path.join(PROJ_ROOT, '.env')
dotenv.load_dotenv(dotenv_path)
images_dpath = os.environ.get()
class Trainer:
    """Class for loading data and defining, compiling, and fitting a model"""

    def __init__(self, config):
        '''
        Filepaths to different data partitions.
        '''
        self.C = config # save sacred config dict
        self.PROJ_ROOT = self.C['PROJ_ROOT']
        self.images_dpath = os.environ.get('images_dpath')


    def load_data(self):
        '''
        Load training and validation data via generators in order to make demands
        on memory manageable.
        '''
        from keras.utils.np_utils import to_categorical
        from keras.preprocessing import image
        from keras.preprocessing.image import ImageDataGenerator
        from keras.applications.resnet50 import preprocess_input
        from sklearn.model_selection import train_test_split
        from image_utilities import load_images_from_directory, preprocess_input_resnet
        import numpy as np

        #load_images from from the train and val directories
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_resnet)
        train_dir = os.path.join(self.images_dpath, 'train')
        self.train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    target_size=[224, 224],
                                                    batch_size=32,
                                                    class_mode='categorical')

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_resnet)
        val_dir = os.path.join(self.images_dpath, 'val')
        self.validation_generator = val_datagen.flow_from_directory(directory=val_dir,
                                                    target_size=[224, 224],
                                                    batch_size=32,
                                                    class_mode='categorical')

    def build_model(self):
        '''
        Each model will have their own architecture.
        '''
        err_str = 'You must implement this method in a subclass defined in models.py!'
        raise NotImplementedError(err_str)

    def compile_model(self):
        '''
        Compile model with optimizer, loss, and metric arguments specified by the
        experiment's configuration.
        '''
        self.model.compile(optimizer=self.C['optimizer'], loss=self.C['loss'], metrics=self.C['metrics'])
        self.model.summary()

    def fit(self):
        '''
        Fit model on training and valdiation generator. Saves outputs to csv,
        tensorboard, mongodb, and model weights.
        '''
        from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint
        from callbacks import AddTimestamp, ErrorMetricsLogger
        from datetime import datetime
        import os
        from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score

        print_params = self.C

        # Name a filepath for saving a csv of model training results
        param_string = '+'.join(["{}={}".format(str(key),str(value)) for key, value in print_params.items()])
        current_time = datetime.now()
        csvLogger_fname = "{}_longtrain@{}.csv".format(self.C['trainer'], current_time)

        # Create folder for holding model_results
        epoch_results_dpath = os.path.join(self.PROJ_ROOT, 'model_results')
        if not os.path.exists(epoch_results_dpath):
            os.makedirs(epoch_results_dpath)
        # Where to save the csv
        epoch_results_fpath = os.path.join(epoch_results_dpath, csvLogger_fname)

        ## Name the path and name of where the model + weights will be saved
        best_model_name = "{}_model@{}".format(self.C['trainer'], current_time)
        saved_models_dir = os.path.join(epoch_results_dpath,'saved_models')
        if not os.path.exists(saved_models_dir):
            os.makedirs(saved_models_dir)
        best_model_fpath = os.path.join(saved_models_dir, best_model_name)
        # name tensorboard logging directory for model results
        tensorboard_dir = os.path.join(epoch_results_dpath, 'tensorboard', 'first_experiment')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        cbs = [ModelCheckpoint(best_model_fpath, monitor='val_loss', mode='min', save_best_only=True),
               AddTimestamp(self.C),
               CSVLogger(epoch_results_fpath, separator=','),
               TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)]

        history = self.model.fit_generator(self.train_generator,
                            steps_per_epoch = self.C['steps_per_epoch'],
                            epochs = self.C['epochs'],
                            validation_data = self.validation_generator,
                            validation_steps = self.C['validation_steps'],
                            callbacks = cbs)

        values = history.history[self.C['metric']]
        best = eval(self.C['result_mode'])

        return best(values)
