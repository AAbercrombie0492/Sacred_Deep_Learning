'''
Define different architectures for experimentation in this script
Each architecture inherits attributes from the Trainer class.
'''
from trainer import Trainer
from keras.regularizers import l2


class ResNet50_FeatureExtractor(Trainer):
    """Pretrained ResNet50 with a softmax classifier that acts
       as a feature extractor. """
    def build_model(self):
        from keras.layers import Dense
        from keras.layers import Activation
        from keras.models import Model
        from resnet50 import ResNet50
        resnet50_model = ResNet50(weights='imagenet')

        fc1000 = resnet50_model.get_layer('fc1000').output
        final_softmax = Dense(output_dim=2, activation='softmax')(fc1000)
        resnet50_ftr_extrctr = Model(input=resnet50_model.input, output=final_softmax)

        for layer in resnet50_ftr_extrctr.layers[:-1]:
            layer.trainable = False

        self.model = resnet50_ftr_extrctr


class ResNet50_FineTune_1skip(Trainer):
    """Pretrained ResNet50 with a softmax classifier that finetunes one skip-batch."""
    def build_model(self):
        from keras.layers import Dense
        from keras.layers import Activation
        from keras.models import Model
        from resnet50 import ResNet50
        from resnet50 import ResNet50

        resnet50_model = ResNet50(weights='imagenet')
        fc1000 = resnet50_model.get_layer('fc1000').output
        final_softmax = Dense(output_dim=2, activation='softmax')(fc1000)
        resnet50_finetune_1skip = Model(input=resnet50_model.input, output=final_softmax)

        for layer in resnet50_finetune_1skip.layers[:-14]:
            layer.trainable = False

        self.model = resnet50_finetune_1skip

class ResNet50_FineTune_2skip(Trainer):
    """Pretrained ResNet50 with a softmax classifier that finetunes two skip-batches."""
    def build_model(self):
        from keras.layers import Dense
        from keras.layers import Activation
        from keras.models import Model
        from resnet50 import ResNet50
        from resnet50 import ResNet50
        resnet50_model = ResNet50(weights='imagenet')

        fc1000 = resnet50_model.get_layer('fc1000').output
        final_softmax = Dense(output_dim=2, activation='softmax')(fc1000)
        resnet50_ftr_extrctr = Model(input=resnet50_model.input, output=final_softmax)

        for layer in resnet50_ftr_extrctr.layers[:-24]:
            layer.trainable = False

        self.model = resnet50_ftr_extrctr

class ResNet50_FineTune_3skip(Trainer):
    """Pretrained ResNet50 with a softmax classifier that finetunes two skip-batches."""
    def build_model(self):
        from keras.layers import Dense
        from keras.layers import Activation
        from keras.models import Model
        from resnet50 import ResNet50
        from resnet50 import ResNet50
        resnet50_model = ResNet50(weights='imagenet')

        fc1000 = resnet50_model.get_layer('fc1000').output
        final_softmax = Dense(output_dim=2, activation='softmax')(fc1000)
        resnet50_ftr_extrctr = Model(input=resnet50_model.input, output=final_softmax)

        for layer in resnet50_ftr_extrctr.layers[:-36]:
            layer.trainable = False

        self.model = resnet50_ftr_extrctr

class ResNet50_FullTrain(Trainer):
    """Pretrained Resnet50 with a softmax classifier where all weights are
    trained on the input data"""
    def build_model(self):
        from keras.layers import Dense
        from keras.layers import Activation
        from keras.models import Model
        from resnet50 import ResNet50
        from resnet50 import ResNet50
        resnet50_model = ResNet50(weights='imagenet')

        fc1000 = resnet50_model.get_layer('fc1000').output
        final_softmax = Dense(output_dim=2, activation='softmax')(fc1000)
        resnet50_ftr_extrctr = Model(input=resnet50_model.input, output=final_softmax)

        self.model = resnet50_ftr_extrctr


class ResNet50_FromScratch(Trainer):
    """Fresh ResNet50 with a softmax classifier with zero initialized weights"""
    def build_model(self):
        from resnet50 import identity_block, conv_block
        from keras.layers import Input
        from keras import layers
        from keras.layers import Dense
        from keras.layers import Activation
        from keras.layers import Flatten
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.layers import GlobalMaxPooling2D
        from keras.layers import ZeroPadding2D
        from keras.layers import AveragePooling2D
        from keras.layers import GlobalAveragePooling2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        from keras.preprocessing import image
        import keras.backend as K
        from keras.utils import layer_utils

        x = ZeroPadding2D((3, 3))(input_shape=self.X[0].shape)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(2, activation='softmax', name = 'fc1000')(x)

        self.model = Model(inputs, x, name='resnet50')


class MLRTrainer(Trainer):
    """Multi-Class Logistic Regression Classifier"""

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Activation

        model = Sequential()
        model.add(Flatten(input_shape=self.X[0].shape))
        model.add(Dense(units=self.y[0].shape[0], W_regularizer=l2(self.C['reg'])))
        model.add(Activation('softmax'))
        self.model = model

class MLPTrainer(Trainer):
    """Multi-Layer Perceptron Classifier"""

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Flatten, Dense, Activation

        model = Sequential()

        model.add(Flatten(input_shape=self.X[0].shape))
        model.add(Dense(units=self.C['hidden_size'], activation='relu'))
        model.add(Dense(units=self.y[0].shape[0]))
        model.add(Activation('softmax'))
        self.model = model


class CNNTrainer(Trainer):
    """Convolutional Neural Network Classifier"""

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten

        model = Sequential()
        model.add(Conv2D(filters=self.C['hidden_size'],
                         kernel_size=(3,3),
                         activation='relu',
                         input_shape=self.X[1].shape))
        model.add(MaxPooling2D(pool_size=self.C['pool_size']))
        model.add(Dropout(rate=self.C['dropout']))
        model.add(Flatten())
        model.add(Dense(self.C['hidden_size'], activation='relu'))
        model.add(Dropout(rate=self.C['dropout']))
        model.add(Dense(units=self.y[0].shape[0]))
        model.add(Dense(units=self.y[0].shape[0], activation='softmax'))

        self.model = model

if __name__ == '__main__':
    from keras.preprocessing import image
    import keras.backend as K
    from keras.utils import layer_utils
    from keras.utils.data_utils import get_file
    from keras.applications.imagenet_utils import decode_predictions
    from keras.applications.imagenet_utils import preprocess_input
    from keras.applications.imagenet_utils import _obtain_input_shape
    from keras.engine.topology import get_source_inputs
    import numpy as np
    from resnet50 import ResNet50

    # feature_extractor = ResNet50_FeatureExtractor()
    # feature_extractor.build_model()
    # model = feature_extractor.model

    model = ResNet50(include_top=True, weights='imagenet')

    img_path = '../../data/satimages/shaded/+36+GERARD+RD+NORWELL+MA.png'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
