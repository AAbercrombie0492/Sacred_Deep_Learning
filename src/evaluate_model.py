"""
Script that loads the weights of the most up to date finetuned ResNet model,
makes predictions on a holdout test set, and evaluates the performance of the
model by a set of error metrics.
"""
import numpy as np
import numpy.random as rnd
import pandas as pd
import os
import sys
import tensorflow as tf
import PIL
import datetime
from sklearn.datasets import load_sample_image
from os import listdir
from os.path import isfile, join

PROJ_ROOT = os.pardir

# add local python functions



from tqdm import tqdm
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from sklearn.cross_validation import train_test_split
from keras.callbacks import TensorBoard, CSVLogger
from keras.models import model_from_json
import random
import pickle
from keras.utils.np_utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
from resnet50 import ResNet50
from imagenet_utils import decode_predictions
from image_utilities import *
import logging



def define_model(weights_path):
    '''
    Define model structure with weights.
    '''
    from resnet50 import ResNet50
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D


    resnet50_model = ResNet50()
    fc1000 = resnet50_model.get_layer('fc1000').output
    final_softmax = Dense(output_dim=2, activation='softmax')(fc1000)
    resnet50_finetune_1skip = Model(input=resnet50_model.input, output=final_softmax)
    resnet50_finetune_1skip.load_weights(weights_path)

    resnet50_finetune_1skip.compile(loss="categorical_crossentropy",
                                optimizer='nadam',
                                metrics=['accuracy'])

    return resnet50_finetune_1skip

def test_images_generator(test_path):
    '''
    Creates a generator that pulls images from a test directory that contains
    shade vs sunny subdirectories.
    '''
    from keras.utils.np_utils import to_categorical
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.resnet50 import preprocess_input
    from sklearn.model_selection import train_test_split
    from image_utilities import load_images_from_directory, preprocess_input_resnet
    import numpy as np

    #load_images from from the train and val directories
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_resnet)
    test_generator = test_datagen.flow_from_directory(directory=test_path,
                                                target_size=[224, 224],
                                                batch_size=26,
                                                class_mode='categorical')

    return test_datagen, test_generator

def make_predictions(test_generator, compiled_model):
    '''
    Make predictions on images from the test_generator and save as a numpy array.
    '''
    import numpy as np
    import pickle
    preds = compiled_model.predict_generator(test_generator, steps=397, workers=10, pickle_safe=True, verbose=1)
    try:
        np.save('predictions_array', preds)
        logging.info('predictions saved')
    except:
        logging.info('predictions not saved')
    return preds

def evaluate_model(preds, test_generator):
    '''
    Evaluate model performance in terms of accuracy, recall, precision, f1, and
    AUC. Stores results in a dictionary and pickles it.
    '''
    from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, classification_report, accuracy_score
    from viz import show_confusion_matrix

    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes
    # y_true = y_true[:len(y_pred)]

    accuracy = accuracy_score(y_true, y_pred)
    logging.info('ACCURACY : {}'.format(accuracy))

    recall = recall_score(y_true, y_pred, average=None).mean()
    logging.info('RECALL : {}'.format(recall))

    precision = precision_score(y_true, y_pred, average=None).mean()
    logging.info('PRECISION : {}'.format(precision))

    auc = roc_auc_score(y_true, y_pred, average=None).mean()
    logging.info('AUC : {}'.format(auc))

    f1 = f1_score(y_true, y_pred, average=None).mean()
    logging.info('F1 : {}'.format(f1))

    C = confusion_matrix(y_true, y_pred)
    logging.info('CONFUSION MATRIX :\n {}'.format(C))

    roc = roc_curve(y_true, y_pred)
    logging.info('ROC : {}'.format(roc))

    report = classification_report(y_true, y_pred)
    logging.info('REPORT :\n {}'.format(report))
    # show_confusion_matrix(C, ['Sunny', 'Shaded'])

    results_dict = {'precision': precision,
                    'recall': recall,
                    'auc': auc,
                    'f1': f1,
                    'confusion_matrix': C,
                    'ROC' : roc,
                    'classification_report' : report,
                    'accuracy': accuracy
                    }
    return results_dict


def main():
    import pickle
    logging.info('DEFINING MODEL AND LOADING BEST WEIGHTS')
    resnet50_finetune_1skip = define_model(best_model_fpath)
    logging.info('CREATING A TESTING DATA GENERATOR')
    test_path = os.path.join(images_dpath, 'test')
    test_datagen, test_generator = test_images_generator(test_path)

    logging.info('MAKE PREDICTIONS')
    preds = make_predictions(test_generator, resnet50_finetune_1skip)
    # preds = np.load('predictions_array.npy')

    logging.info('EVALUATION MODEL')
    results_dict = evaluate_model(preds, test_generator)

    logging.info('STORING MODEL RESULTS @ model_results.p')
    with open('model_results.p', 'wb') as f:
        pickle.dump(results_dict, f)

    logging.info('SUCCESS')

if __name__ == '__main__':
    log = logging.INFO
    logging.basicConfig(level=log)

    import dotenv

    PROJ_ROOT = os.pardir
    dotenv_path = os.path.join(PROJ_ROOT, '.env')
    dotenv.load_dotenv(dotenv_path)
    images_dpath = os.environ.get('images_dpath')

    logging.info('DEFINING FILEPATHS')
    images_dpath = os.environ.get('images_dpath')

    from keras.models import model_from_json
    best_model_fpath = os.path.join(PROJ_ROOT,
                                    'data',
                                    'model_results',
                                    'saved_models',
                                    'ResNet50_FineTune_1skip_model@2017-05-02 21:02:05.198958'
                                    )

    main()
