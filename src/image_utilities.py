"""
Contains modules for loading, plotting, and processsing image data.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import tqdm

def save_fig(fig_id, tight_layout=True):
    '''
    Save figure to file.
    '''
    path = os.path.join(PROJ_ROOT, "visualizations", NOTEBOOK_ID, fig_id + ".png")
    if not os.path.exists(path):
        os.makedirs(path)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_image(image):
    '''
    Plot image array in grayscale.
    '''
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    '''
    Plot image array in color.
    '''
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")

def PIL2array(img):
    '''
    Converts a PIL object to a numpy array.
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    '''
    Converts a numpy array to a PIL object.
    '''
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def load_images_from_directory(directory, n_samples):
    '''
    Load n number of images from a directory.
    Returns image arrays, filenames, and a fail log.
    '''
    from os import listdir
    from os.path import isfile, join
    from keras.preprocessing import image
    from tqdm import tqdm
    from keras.applications.resnet50 import preprocess_input

    fail_log = ''
    image_arrays = []
    files = [f for f in listdir(directory) if isfile(join(directory, f))]

    for img_path in tqdm(files[:n_samples]):
        # try:
        full_path = os.path.join(directory, img_path)
        img = image.load_img(full_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        image_arrays.append(x)
        # except:
            # fail_log +='{}\n'.format(img_path)
            # continue

    return image_arrays, files, fail_log

def preprocess_input_resnet(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass keras.applications.resnet50.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).
    """
    import numpy as np
    from keras.applications.resnet50 import preprocess_input
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x[0]
