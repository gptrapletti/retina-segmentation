import monai
import pytorch_lightning as pl
import torch

class RetinaUNet(pl.LightningModule):
    def __init__(self, backbone, loss_function, metric, optimizer, scheduler):
        super().__init__()
        self.backbone = backbone
        self.loss_function = loss_function
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, gts = batch['image'], batch['mask']
        preds = self(images)
        preds_threshold = torch.where(preds > 0.5, 1.0, 0.0)
        preds_threshold.grad = preds.grad
        loss = self.loss_function(preds_threshold, gts)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, gts = batch['image'], batch['mask']
        preds = self(images)
        preds_threshold = torch.where(preds > 0.5, 1.0, 0.0)
        preds_threshold.grad = preds.grad
        loss = self.loss_function(preds_threshold, gts)
        self.log("val_loss", loss, prog_bar=True)
        metric = self.metric(
            preds, 
            gts.clone().detach().astype(torch.int8)
        )
        self.log("val_metric", metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, gts = batch['image'], batch['mask']
        preds = self(images)
        metric = self.metric(
            preds, 
            gts.clone().detach().astype(torch.int8)
        )
        self.log("test_metric", metric, prog_bar=True)

    def configure_optimizers(self):
        optim = self.optimizer
        sched = self.scheduler
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}