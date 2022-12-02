# main.py
import torch
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl
# simple demo classes for your convenience
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule
from svtr_ocr.config import read_config
from svtr_ocr.model import _create_svtr

class MyModel(pl.LightningModule):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = read_config(config_path)
        self.save_hyperparameters()

        self.model = _create_svtr(
            cfg_name=self.config.model.name, 
            img_size=self.config.model.image_size,
            in_channels=self.config.model.channels,
            out_channels=len(self.config.charset.chars) + 1,
        )
        self.loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    
    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch, batch_idx, mode):
        print(batch)
        return 0
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, config_path: str):
        super().__init__()
        self.save_hyperparameters()

def cli_main():
    LightningCLI(MyModel, MyDataModule, run=True)
    

if __name__ == "__main__":
    cli_main()
