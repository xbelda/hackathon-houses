from pathlib import Path

import torch
from torchvision import datasets, transforms
from src.model import HouseNet
import pytorch_lightning as pl


BASE_PATH = Path("./data/House_Rooms_Images_Split")
INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8
LEARNING_RATE = 1e-3
FREEZE_CNN = True
RESNET_SIZE = 34



def main():
    # Train Data
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(BASE_PATH / "train", transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   num_workers=NUM_WORKERS,
                                                   shuffle=True)
    # Test Data
    test_transforms = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(BASE_PATH / "test", transform=test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=BATCH_SIZE,
                                                  num_workers=NUM_WORKERS,
                                                  shuffle=False)

    # Ensure that all classes are correctly set
    assert train_dataset.classes == test_dataset.classes

    # Model
    pt_model = HouseNet(num_classes=len(train_dataset.classes),
                        lr=LEARNING_RATE,
                        freeze_cnn=FREEZE_CNN,
                        resnet_size=RESNET_SIZE)

    # Early Stopping
    early_stopping = pl.callbacks.EarlyStopping('Loss/Val', min_delta=1e-3)

    # Logging
    logger = pl.loggers.MLFlowLogger()

    # Train
    trainer = pl.Trainer(gpus=torch.cuda.device_count(), callbacks=[early_stopping], logger=logger)
    trainer.logger.log_hyperparams({"batch_size": BATCH_SIZE})
    trainer.fit(pt_model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
