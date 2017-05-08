'''
This script defines hyperparameters, stores them in a Sacred config object,
and passes that object to the trainer.py module which builds and compiles the
model.

Options for running from the command line:
- ```python run_experiment.py print_config```
- ```python run_experiment.py with optimizer=adam```
'''
from models import *
from sacred import Experiment
from callbacks import ErrorMetricsLogger
ex = Experiment()

@ex.config
def my_config():
    trainer = 'ResNet50_FineTune_1skip' # architecture
    loss = 'categorical_crossentropy' # type of loss
    # hidden_size = 128 # number of units in hidden layer
    # pool_size = (2,2)
    # reg = 1e-6
    epochs = 5
    # dropout = 0.5
    frontend = 'cmd' # set to 'cmd' when calling from the command line!
    optimizer = 'nadam'
    metrics = ['accuracy']
    metric = 'val_loss'
    result_mode = 'min'
    n_samples = 10
    test_size = 0.1
    PROJ_ROOT = '/home/ubuntu/PS_shading_model/'
    steps_per_epoch = 2
    validation_steps = 1

# TO RUN FROM COMMMAND line
# python run_experiment.py print_config
# python run_experiment.py with optimizer=adam

@ex.automain
def main(_config, _run):
    """Run a sacred experiment

    Parameters
    ----------
    _config : special dict populated by sacred with the local variables computed
    in my_config() which can be overridden from the command line or with
    ex.run(config_updates=<dict containing config values>)
    _run : special object passed in by sacred which contains (among other
    things) the name of this run


    This function will be run if this script is run either from the command line
    with

    $ python run_experiment.py

    or from within python by

    >>> from run_experiment import ex
    >>> ex.run()

    """
    _config['name'] = _run.meta_info['options']['--name']

    trainer = eval(_config['trainer'])(_config)
    trainer.load_data()
    trainer.build_model()
    trainer.compile_model()
    result = trainer.fit()

    return result
