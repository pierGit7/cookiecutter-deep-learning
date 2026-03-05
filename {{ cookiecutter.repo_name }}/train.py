import os
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    print(f"Starting training with framework: {cfg.framework}")
    
{%- if cookiecutter.dl_framework == 'keras3' %}
    # Set backend before importing keras
    if "backend" in cfg:
        os.environ["KERAS_BACKEND"] = cfg.backend
        
    import keras
    from {{ cookiecutter.module_name }}.models.architecture import build_model
    from {{ cookiecutter.module_name }}.data.dataset import get_dataset
    
    model = build_model(cfg.model)
    train_dataset = get_dataset(cfg.data, split="train")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.training.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = []
    {%- if cookiecutter.experiment_tracker == 'wandb' %}
    from wandb.keras import WandbCallback
    import wandb
    wandb.init(project="{{ cookiecutter.project_name }}", config=dict(cfg))
    callbacks.append(WandbCallback())
    {%- elif cookiecutter.experiment_tracker == 'tensorboard' %}
    callbacks.append(keras.callbacks.TensorBoard(log_dir="./logs"))
    {%- endif %}
    
    model.fit(
        train_dataset,
        epochs=cfg.training.epochs,
        callbacks=callbacks
    )

{%- elif cookiecutter.dl_framework == 'pytorch' %}
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from {{ cookiecutter.module_name }}.models.architecture import build_model
    from {{ cookiecutter.module_name }}.data.dataset import get_dataloader

    # PyTorch Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    train_loader = get_dataloader(cfg.data, split="train")
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.CrossEntropyLoss()
    
    {%- if cookiecutter.experiment_tracker == 'wandb' %}
    import wandb
    wandb.init(project="{{ cookiecutter.project_name }}", config=dict(cfg))
    {%- elif cookiecutter.experiment_tracker == 'tensorboard' %}
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="./logs")
    {%- endif %}
    
    for epoch in range(cfg.training.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            {%- if cookiecutter.experiment_tracker == 'wandb' %}
            wandb.log({"train_loss": loss.item()})
            {%- elif cookiecutter.experiment_tracker == 'tensorboard' %}
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)
            {%- endif %}
            
{%- elif cookiecutter.dl_framework == 'tensorflow' %}
    import tensorflow as tf
    from {{ cookiecutter.module_name }}.models.architecture import build_model
    from {{ cookiecutter.module_name }}.data.dataset import get_dataset

    # TensorFlow Configuration
    model = build_model(cfg.model)
    train_dataset = get_dataset(cfg.data, split="train")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    callbacks = []
    {%- if cookiecutter.experiment_tracker == 'wandb' %}
    import wandb
    from wandb.keras import WandbCallback
    wandb.init(project="{{ cookiecutter.project_name }}", config=dict(cfg))
    callbacks.append(WandbCallback())
    {%- elif cookiecutter.experiment_tracker == 'tensorboard' %}
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="./logs"))
    {%- endif %}
    
    model.fit(
        train_dataset,
        epochs=cfg.training.epochs,
        callbacks=callbacks
    )
{%- endif %}

if __name__ == "__main__":
    main()
