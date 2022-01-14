from typing import Dict

import numpy as np
import tensorflow as tf

from github_repo.transformations import normalize_01

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_function(input_payload: dict, model: tf.keras.Model, input_parameters: Dict) -> np.ndarray:
    # preprocessing
    image = np.array([input_payload])  # convert to numpy array
    image = normalize_01(image)  # linear scaling to range [0-1]
    image = image.astype(np.float32)  # typecasting to float32

    return image


def postprocess_function(output: np.ndarray) -> np.ndarray:
    # postprocessing
    out = np.argmax(output)  # perform argmax to generate 1 channel
    out = class_names[out]  # change class index to class name
    return np.array([out])
