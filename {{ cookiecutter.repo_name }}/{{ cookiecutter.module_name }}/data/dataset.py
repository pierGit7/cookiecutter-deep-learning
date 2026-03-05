import numpy as np

{%- if cookiecutter.dl_framework == 'keras3' or cookiecutter.dl_framework == 'tensorflow' %}
import tensorflow as tf

def get_dataset(cfg, split="train"):
    """Returns a tf.data.Dataset for Keras/TensorFlow."""
    # Placeholder: replace with actual data loading
    num_samples = 1000 if split == "train" else 200
    x = np.random.randn(num_samples, cfg.input_dim).astype(np.float32)
    y = np.random.randint(0, cfg.num_classes, size=(num_samples,))
    
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    if split == "train":
        dataset = dataset.shuffle(1024)
        
    return dataset.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

{%- elif cookiecutter.dl_framework == 'pytorch' %}
import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, num_samples, input_dim, num_classes):
        self.x = torch.randn(num_samples, input_dim)
        self.y = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloader(cfg, split="train"):
    """Returns a PyTorch DataLoader."""
    num_samples = 1000 if split == "train" else 200
    dataset = DummyDataset(num_samples, cfg.input_dim, cfg.num_classes)
    
    return DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=(split == "train"),
        num_workers=cfg.num_workers
    )
{%- endif %}
