from pathlib import Path

import torch
from torchvision import datasets, transforms
from src.model import HouseNet
import pytorch_lightning as pl
from omegaconf import DictConfig


def train_net(config: DictConfig):
    base_path = Path(config.BASE_PATH)
    input_size = config.INPUT_SIZE
    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS
    learning_rate = config.LEARNING_RATE
    freeze_cnn = config.FREEZE_CNN
    resnet_size = config.RESNET_SIZE

    # Train Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.ImageFolder(base_path / "train", transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True)
    # Test Data
    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = datasets.ImageFolder(base_path / "test", transform=test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  shuffle=False)

    # Ensure that all classes are correctly set
    assert train_dataset.classes == test_dataset.classes

    # Model
    pt_model = HouseNet(num_classes=len(train_dataset.classes),
                        lr=learning_rate,
                        freeze_cnn=freeze_cnn,
                        resnet_size=resnet_size)

    # Early Stopping
    early_stopping = pl.callbacks.EarlyStopping('Loss/Val', min_delta=1e-3)

    # Logging
    logger = pl.loggers.MLFlowLogger(experiment_name="houses",
                                     run_name="2",
                                     tags={"status": "experimentation",
                                           "dataset": 123})

    # Train
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         max_epochs=50,
                         callbacks=[early_stopping],
                         logger=logger)
    trainer.logger.log_hyperparams({"batch_size": batch_size})
    trainer.fit(pt_model, train_dataloader, test_dataloader)
