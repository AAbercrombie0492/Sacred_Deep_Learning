# Sacred_Deep_Learning
Infrastructure for iterative and organized experimentation with CNN image classification.
==============================

Configuring neural network architectures and optimizing hyperparameters is an
iterative processes that is prone to becoming quite messy.

<p><a target="_blank" href="https://github.com/IDSIA/sacred">Sacred</a>
is a framework for configuring and executing experiments that can be safely
stored in a Mongo Database.</p>

Since Neural Networks can take a long time to train, it's also nice to be able
to monitor and visualize training progress with tools such as <a target="_blank" href="https://www.tensorflow.org/get_started/summaries_and_tensorboard">TensorBoard</a>
and even <a target="_blank" href="https://www.tensorflow.org/get_started/summaries_and_tensorboard">SacredBoard</a>

This repo demonstrates how to create a system that implements these tools in the
context of training an image classifier in Keras.
See the notebook ```PROJ_ROOT/notebooks/Sacred_Experiments.ipynb``` for an implementation example.

I do not provide any data, but I would recommend testing this out with <a target="_blank" href="https://www.kaggle.com/c/dogs-vs-cats">Cats Vs Dogs.</a> You'll notice that
my code references environment variables stored in an .env file. NEVER ADD YOUR .env FILE TO VERSION CONTROL!

Here's how you can identify your own .env variables for easy reproducibility:

```bash
image_dpath = /ubuntu/users/Sacred_Deep_Learning/data/images
train_dpath = /ubuntu/users/Sacred_Deep_Learning/data/images/train
test_dpath = /ubuntu/users/Sacred_Deep_Learning/data/images/test
val_dpath = /ubuntu/users/Sacred_Deep_Learning/data/images/val


```

Here's how you load environment variables within a Python script with the python-dotenv package.
```python
# Environment variables go here, can be read by `python-dotenv` package:
#
#   `src/script.py`
#   ----------------------------------------------------------------
#    import dotenv
#
#    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
#    dotenv_path = os.path.join(project_dir, '.env')
#    dotenv.load_dotenv(dotenv_path)
#    AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
#   ----------------------------------------------------------------
```

To setup:
```bash
$ conda env create -f environment.yml #activate the environment I provided
$ brew install mongodb # install mongodb
$ mkdir {PROJ_ROOT}/model_results/mongo # create local directory for mongodb to write to
$ mongod --dbpath mongo # start mongodb server and tell it to write to local folder mongo
$ pip install git+https://github.com/IDSIA/sacred.git # install latest version of sacred
$ pip install sacredboard # install sacredboard
$ sacredboard # start a default sacredboard server
$ sacredboard experiment5_architectures # start a sacredboard server that references a mongodb table
```

# Module Overview
==============================

```src/driver.py```
------------
Setup and execute sacred experiments that are saved in a specified mongodb tables. Model updates are stored in:
- ```PROJ_ROOT/model_results/mongo```
- ```PROJ_ROOT/model_results/saved_models```
- ```PROJ_ROOT/model_results/tensorboard```
- ```PROJ_ROOT/model_results/csv_results```


To access results:
------------
- Setup a TensorBoard server with the following command:
  - ```tensorboard --logdir PROJ_ROOT/model_results/tensorboard```
  - Navigate to localhost:6000 in a web browser.
- The Keras callback CSVLogger is used to store training results in ```PROJ_ROOT/model_results/csv_results```
- Setup a SacredBoard server with ```sacredboard {NAME_OF_EXPERIMENT}
- The Keras callback ModelCheckpoints stores HDF5 weights every epoch in ```PROJ_ROOT/model_results/saved_models```

```src/trainer.py```
------------
Parent class that will be inherited by each specific trainer.
Takes care of data loading/processing, model compiling, and fitting.

```src/models.py```
------------
Define different architectures for experimentation in this script
Each architecture inherits attributes from the Trainer class.

```src/resnet50.py```
------------
ResNet50 model for Keras written by Francois Chollet.
### Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan and FChollet.

```src/callbacks.py```
------------
Callback modules that are executed after each training epoch.

```src/run_experiment.py```
------------
This script defines hyperparameters, stores them in a Sacred config object,
and passes that object to the trainer.py module which builds and compiles the
model.

Options for running from the command line:
- ```python run_experiment.py print_config```
- ```python run_experiment.py with optimizer=adam```

```src/evaluate_model.py```
------------
Script that loads the weights of the most up to date finetuned ResNet model,
makes predictions on a holdout test set, and evaluates the performance of the
model by a set of error metrics.

```src/image_utilities.py```
------------
Modules for loading, transforming, processing, and visualizing images.
