"""
Callback modules that are executed after each training epoch.
"""

from keras.callbacks import Callback
from datetime import datetime
import scipy
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import pickle
from keras.utils.np_utils import to_categorical
import keras.backend as K
from collections import defaultdict


class AddTimestamp(Callback):
    """Assigns unique ids to metrics for scalar summary visualizations in TensorBoard"""

    def __init__(self, config):
        # self.exp = '+'.join([f'{key}={value}' for key, value in exp.items()])
        # name = config
        # self.exp = '+'.join(["{}={}".format(str(key),str(value)) for key, value in exp.items()])
        # # self.exp = '+'.join([f"{key}={value}" for key, value in exp.items()])
        # self.T = datetime.now()
        name = config['name']
        self.exp = '+'.join(f'{key}={value}' for key, value in config.items()) if not name else name
        self.T = datetime.now()

    def on_epoch_end(self, epoch, logs={}):
        for metric in self.params['metrics']:
            # logs[f'{metric}:{self.exp}@{self.T}'] = logs[metric]
            logs["{}:{}@{}".format(str(metric), str(self.exp), str(self.T))] = logs[metric]

class F1Logger(Callback):
    """Calculates F1 Score for each epoch and stores in logs"""

    def __init__(self, X_val, y_val):
        import numpy as np
        # initiative Callback from parent class
        super(Callback, self).__init__()
        self.X_val, self.y_val = np.array(X_val), np.argmax(y_val)
    def on_epoch_end(self, epoch, logs={}):
        import numpy as np
        from sklearn.metrics import f1_score
        # y_pred = np.array(self.model.predict(self.X_val).argmax())
        y_pred = self.model.predict(self.X_val)
        y_pred = np.argmax(y_pred)
        f1s = f1_score(self.y_val, y_pred, average=None)
        f1_score = np.mean(f1s)
        logs['f1_score'].append(f1_score)

class AUCLogger(Callback):
    """Calculates AUC Score for each epoch and stores in logs"""

    def __init__(self, X_val, y_val):
        import numpy as np
        # initiative Callback from parent class
        super(Callback, self).__init__()
        self.X_val, self.y_val = np.array(X_val), np.array(y_val)
    def on_epoch_end(self, epoch, logs={}):
        import numpy as np
        from sklearn.metrics import roc_auc_score
        # y_pred = np.array(self.model.predict(self.X_val).argmax())
        y_pred = self.model.predict(self.X_val)
        y_pred = np.array(y_pred)
        auc = roc_auc_score(self.y_val, y_pred, average=None)
        logs['auc_score'].append(auc)

class PrecisionLogger(Callback):
    """Calculates AUC Score for each epoch and stores in logs"""

    def __init__(self, X_val, y_val):
        import numpy as np
        # initiative Callback from parent class
        super(Callback, self).__init__()
        self.X_val, self.y_val = np.array(X_val), np.array(y_val)
    def on_epoch_end(self, epoch, logs={}):
        import numpy as np
        from sklearn.metrics import precision_score
        # y_pred = np.array(self.model.predict(self.X_val)).argmax()
        y_pred = self.model.predict(self.X_val)
        y_pred = np.array(y_pred)
        precision = precision_score(self.y_val, y_pred, average=None)
        logs['precision_score'].append(precision)


class RecallLogger(Callback):
    """Calculates Error Metrics for each epoch and stores in logs"""

    def __init__(self, X_val, y_val):
        import numpy as np
        # initiative Callback from parent class
        super(Callback, self).__init__()
        self.X_val, self.y_val = np.array(X_val), np.array(y_val)

    def on_epoch_end(self, epoch, logs={}):
        import numpy as np
        from sklearn.metrics import recall_score
        # y_pred = np.array(self.model.predict(self.X_val)).argmax()
        y_pred = self.model.predict(self.X_val)
        y_pred = np.array(y_pred)
        recall = recall_score(self.y_val, y_pred, average=None)
        logs['recall_score'].append(recall)

class ErrorMetricsLogger(Callback):
    '''
    Calculates and stores recall, precision, AUC, and F1 all in one pass.
    '''
    def __init__(self, X_val, y_val):
        import numpy as np
        # initiative Callback from parent class
        super(Callback, self).__init__()
        self.X_val, self.y_val = np.array(X_val), np.argmax(y_val, axis=1)

    def on_train_begin(self, logs={}):
        self.recall = []
        self.precision = []
        self.auc = []
        self.f1 = []

    def on_epoch_end(self, epoch, logs={}):
        import numpy as np
        from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
        y_pred = self.model.predict(self.X_val)
        y_pred = np.argmax(y_pred, axis=1)

        recall = recall_score(self.y_val, y_pred, average=None).mean()
        self.recall.append(recall)
        logs['recall'] = recall

        precision = precision_score(self.y_val, y_pred, average=None).mean()
        self.precision.append(precision)
        logs['precision'] = precision

        auc = roc_auc_score(self.y_val, y_pred, average=None).mean()
        self.auc.append(auc)
        logs['auc'] = auc

        f1 = f1_score(self.y_val, y_pred, average=None).mean()
        self.f1.append(f1)
        logs['f1'] = f1
