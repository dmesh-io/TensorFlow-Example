# This file instantiates the model
from typing import Dict

import tensorflow as tf


def get_model(weights_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(weights_path)
