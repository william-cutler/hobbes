import os
import sys

module_path = os.path.abspath(os.path.join('/home/grail/willaria_research/hobbes/calvin_models'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pytorch_lightning as pl
from torch.nn import functional as F
from torch import nn
import torch
from typing import Dict, Tuple
from calvin_agent.models.perceptual_encoders.vision_network_gripper import VisionNetwork as GripperVisionNetwork
from calvin_agent.models.perceptual_encoders.vision_network import VisionNetwork as StaticVisionNetwork



class SingleTaskImitator(pl.LightningModule):
    def __init__(self, rgb_static=False, depth_static=False, rgb_gripper=False, depth_gripper=False, proprioceptive=False, rgb_tactile=False, depth_tactile=False):
        super().__init__()
        # for 'rgb_static' camera
        # input: [N, 3, 200, 200]
        # output: [N, 64]       
        self.static_vision_encoder = None
        self.gripper_vision_encoder = None
        self.proprioceptive_encoder = None
         
        static_channels = 0
        static_output_size = 0
        if rgb_static:
            static_channels += 3
            static_output_size += 64

        if depth_static:
            static_channels += 1
            static_output_size += 32


        if static_channels > 0:
            self.static_vision_encoder = StaticVisionNetwork(200, 200, 'LeakyReLU', 0.0, True, static_output_size, static_channels)

        # for 'rgb_gripper' camera
        # input: [N, 3, 84, 84]
        # output: [N, 32]
        gripper_channels = 0
        gripper_output_size = 0
        if rgb_gripper:
            gripper_channels += 3
            gripper_output_size += 32

        if depth_gripper:
            gripper_channels += 1
            gripper_output_size += 16


        if gripper_channels > 0:
            self.gripper_vision_encoder = GripperVisionNetwork('nature_cnn', 'LeakyReLU', 0.0, False, gripper_output_size, gripper_channels)

        # for proprioceptive information ('robot_obs')
        proprioceptive_output_size = 0
        if proprioceptive:
            self.proprioceptive_encoder = nn.Identity()
            proprioceptive_output_size += 15


        self.ff = nn.Sequential(
            nn.Linear(static_output_size + gripper_output_size + proprioceptive_output_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 7),
            nn.Tanh(),
        )
        # NOTE: gripper (final value in action) is forced to -1 or 1 binary at evaluation time

    def forward(self, imgs: Dict[str, torch.Tensor]):
        """_summary_

        Args:
                imgs (Dict[str, torch.Tensor]): {'static': ..., 'gripper': ..., 'proprioceptive': ...}

        Returns:
                _type_: _description_
        """
        all_embeddings = []
        
        if self.static_vision_encoder:
            static_vision_embeddings = self.static_vision_encoder(imgs['static'])
            all_embeddings.append(static_vision_embeddings)
            
        if self.gripper_vision_encoder:
            gripper_vision_embeddings = self.gripper_vision_encoder(imgs['gripper'])
            all_embeddings.append(gripper_vision_embeddings)
            
        if self.proprioceptive_encoder:
            robot_embeddings = self.proprioceptive_encoder(imgs['proprioceptive'])
            all_embeddings.append(robot_embeddings)
        
        all_embeddings = torch.cat(all_embeddings, dim=-1)
        action = self.ff(all_embeddings)
        return action

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self,
                      train_batch: Tuple[Dict, torch.Tensor],
                      batch_idx: int,
                      ) -> torch.Tensor:
        """_summary_

        Args:
                train_batch (tuple): ({'static': ..., 'gripper': ...}, robot_obs)
                batch_idx (int): _description_

        Returns:
                torch.Tensor: _description_
        """
        x, y = train_batch

        y_hat = self(x)
        
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    # NOTE: Not currently being used
    def test_step(self,
                  train_batch: Tuple[Dict, torch.Tensor],
                  batch_idx: int,
                  ) -> torch.Tensor:
        x, y = train_batch

        y_hat = self(x)
        
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def validation_step(self,
                        train_batch: Tuple[Dict, torch.Tensor],
                        batch_idx: int,
                        ) -> torch.Tensor:
        x, y = train_batch

        y_hat = self(x)
        
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
