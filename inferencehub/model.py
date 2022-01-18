# This file instantiates the model
from typing import Dict

import tensorflow as tf


class ModelWrapper(tf.keras.Model):

    def __init__(self, weights_path: str, parameters: Dict = None, device: str = None):
        super().__init__()
        self.model = tf.keras.models.load_model(weights_path)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x) -> tf.Tensor:
        print(x)
        out = self.model(x)
        return out


def get_model(weights_path: str = None, map_location="cpu",
              model_initialization_parameters: Dict = None) -> tf.keras.Model:
    return ModelWrapper(weights_path)
