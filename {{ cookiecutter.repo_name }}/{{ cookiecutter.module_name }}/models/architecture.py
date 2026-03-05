{%- if cookiecutter.dl_framework == 'keras3' %}
import keras
from keras import layers

def build_model(cfg):
    """Builds a Keras 3 model based on configuration."""
    model = keras.Sequential([
        layers.Input(shape=(cfg.input_dim,)),
        layers.Dense(cfg.hidden_dim, activation='relu'),
        layers.Dense(cfg.output_dim, activation='softmax')
    ])
    return model
{%- elif cookiecutter.dl_framework == 'pytorch' %}
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def build_model(cfg):
    """Builds a PyTorch model based on configuration."""
    return SimpleModel(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
{%- elif cookiecutter.dl_framework == 'tensorflow' %}
import tensorflow as tf
from tensorflow.keras import layers

def build_model(cfg):
    """Builds a TensorFlow model based on configuration."""
    model = tf.keras.Sequential([
        layers.Input(shape=(cfg.input_dim,)),
        layers.Dense(cfg.hidden_dim, activation='relu'),
        layers.Dense(cfg.output_dim, activation='softmax')
    ])
    return model
{%- endif %}
