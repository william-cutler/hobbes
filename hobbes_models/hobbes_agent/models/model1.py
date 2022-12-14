import os
import sys

module_path = os.path.abspath(os.path.join('/home/grail/willaria_research/hobbes/calvin_models'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pytorch_lightning as pl
from torch.nn import functional as F
from torch import nn
import torch
from calvin_agent.models.perceptual_encoders.vision_network_gripper import VisionNetwork as GripperVisionNetwork
from calvin_agent.models.perceptual_encoders.vision_network import VisionNetwork as StaticVisionNetwork


class Model1(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # input: [N, 3, 200, 200]
        # output: [N, 64]
        self.vision_encoder = StaticVisionNetwork(200, 200, 'LeakyReLU', 0.0, True, 64, 3)
        self.ff = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 7),
            nn.Tanh(),
        )
        # NOTE: gripper (final value in action) is forced to -1 or 1 binary at evaluation time

    def forward(self, x):
        embedding = self.vision_encoder(x)
        action = self.ff(embedding)
        return action

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    # NOTE: Not currently being used
    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
