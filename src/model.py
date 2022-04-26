import pytorch_lightning as pl
import torch
from torchvision import models
import timm


class HouseNet(pl.LightningModule):

    def __init__(self,
                 num_classes: int,
                 lr: float,
                 freeze_cnn: bool = True,
                 resnet_size: int = 34):
        super().__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr

        if resnet_size == 18:
            self.model = models.resnet18(pretrained=True)
        elif resnet_size == 34:
            self.model = models.resnet34(pretrained=True)
        elif resnet_size == 50:
            self.model = timm.create_model("resnext50d_32x4d", pretrained=True)

        # Change classification layer
        num_in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_in_features, self.num_classes)

        # Freeze to finetune
        if freeze_cnn:
            self.freeze_backbone()

        self.loss_fc = torch.nn.CrossEntropyLoss()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # in lightning, forward defines the prediction/inference actions
        return self.model(batch)

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def _base_step(self, batch):
        images, y_true = batch
        y_pred = self(images)

        loss = self.loss_fc(y_pred, y_true)
        accuracy = (y_pred.argmax(dim=1) == y_true).float().mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, accuracy = self._base_step(batch)
        self.log("Loss/Train", loss)
        self.log("Accuracy/Train", accuracy)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, accuracy = self._base_step(batch)
        self.log("Loss/Val", loss)
        self.log("Accuracy/Val", accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "Loss/Val",
            "frequency": 10
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
