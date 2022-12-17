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
from bert_language_encoder import BertLanguageEncoder
from vgg_vision_encoder import VGGVisionEncoder



class MultiTaskModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # for 'rgb_static' camera
        # input: [N, 3, 200, 200]
        # output: [N, 64]
        self.static_vision_encoder = StaticVisionNetwork(200, 200, 'LeakyReLU', 0.0, True, 64, 3) #nn.Linear(25088, 64)

        # for 'rgb_gripper' camera
        # input: [N, 3, 84, 84]
        # output: [N, 32]
        self.gripper_vision_encoder = GripperVisionNetwork('nature_cnn', 'LeakyReLU', 0.0, False, 32, 3) #nn.Linear(25088, 32)

        # for proprioceptive information ('robot_obs')
        # output: [N, 15]
        self.robot_encoder = nn.Identity()

        # input: String (sentence)
        # output: [N, 64]
        self.language_encoder = nn.Linear(768, 64)

        
        feature_concat_size = 64 + 32 + 15 + 64
        
        self.ff = nn.Sequential(
            nn.Linear(feature_concat_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 7),
            nn.Tanh(),
        )
        # NOTE: gripper (final value in action) is forced to -1 or 1 binary at evaluation time

    def forward(self, imgs: Dict[str, torch.Tensor], robot_obs: torch.Tensor, language_instruction: str):
        """_summary_

        Args:
                imgs (Dict[str, torch.Tensor]): {'rgb_static': ..., 'rgb_gripper': ...}
                robot_obs (torch.Tensor): _description_

        Returns:
                _type_: _description_
        """
        static_vision_embeddings = self.static_vision_encoder(imgs['rgb_static'])
        gripper_vision_embeddings = self.gripper_vision_encoder(imgs['rgb_gripper'])
        robot_embeddings = self.robot_encoder(robot_obs)
        
        
        lang_embedding = self.language_encoder(language_instruction)

        all_embeddings = torch.cat([static_vision_embeddings, gripper_vision_embeddings, robot_embeddings, lang_embedding], dim=-1)
        action = self.ff(all_embeddings)
        return action

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self,
                      train_batch: Tuple[Tuple[Dict, torch.Tensor, str], torch.Tensor],
                      batch_idx: int,
                      ) -> torch.Tensor:
        """_summary_

        Args:
                train_batch (tuple): ({'rgb_static': ..., 'rgb_gripper': ...}, robot_obs)
                batch_idx (int): _description_

        Returns:
                torch.Tensor: _description_
        """
        x, y = train_batch
        imgs, robot_obs, lang = x

        y_hat = self(imgs, robot_obs, lang)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    # NOTE: Not currently being used
    def test_step(self,
                  train_batch: Tuple[Tuple[Dict, torch.Tensor, str], torch.Tensor],
                  batch_idx: int,
                  ) -> torch.Tensor:
        x, y = train_batch
        imgs, robot_obs, lang = x

        y_hat = self(imgs, robot_obs, lang)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def validation_step(self,
                  train_batch: Tuple[Tuple[Dict, torch.Tensor, str], torch.Tensor],
                  batch_idx: int,
                  ) -> torch.Tensor:
        x, y = train_batch
        imgs, robot_obs, lang = x

        y_hat = self(imgs, robot_obs, lang)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss