import os
import sys
module_path = os.path.abspath(os.path.join('/home/grail/willaria_research/hobbes/calvin_models'))
if module_path not in sys.path:
    sys.path.append(module_path)


from calvin_agent.models.perceptual_encoders.vision_network import VisionNetwork as StaticVisionNetwork
from calvin_agent.models.perceptual_encoders.vision_network_gripper import VisionNetwork as GripperVisionNetwork
from typing import Dict, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl

class Stage1Model(pl.LightningModule):
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
			nn.Sigmoid(),
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






class Stage1ModelV2(pl.LightningModule):
	def __init__(self):
		super().__init__()
		# for 'rgb_static' camera
		# input: [N, 3, 200, 200]
		# output: [N, 64]
		self.static_vision_encoder = StaticVisionNetwork(200, 200, 'LeakyReLU', 0.0, True, 64, 3)

		# for 'rgb_gripper' camera
		# input: [N, 3, 84, 84]
		# output: [N, 32]
		self.gripper_vision_encoder = GripperVisionNetwork('nature_cnn', 'LeakyReLU', 0.0, False, 64, 3)

		# for proprioceptive information ('robot_obs')
		self.robot_encoder = nn.Identity()

		self.ff = nn.Sequential(
			nn.Linear(143, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 64),
			nn.LeakyReLU(),
			nn.Linear(64, 7),
			nn.Sigmoid(),
		)
		# NOTE: gripper (final value in action) is forced to -1 or 1 binary at evaluation time


	def forward(self, imgs: Dict[str, torch.Tensor], robot_obs: torch.Tensor):
		"""_summary_

		Args:
			imgs (Dict[str, torch.Tensor]): {'rgb_static': ...,
											 'rgb_gripper': ...}
			robot_obs (torch.Tensor): _description_

		Returns:
			_type_: _description_
		"""
		static_vision_embeddings = self.static_vision_encoder(imgs['rgb_static'])
		gripper_vision_embeddings = self.gripper_vision_encoder(imgs['rgb_gripper'])
		robot_embeddings = self.robot_encoder(robot_obs)
		# print(static_vision_embeddings.shape)
		# print(gripper_vision_embeddings.shape)
		# print(robot_embeddings.shape)
		all_embeddings = torch.cat([static_vision_embeddings, gripper_vision_embeddings, robot_embeddings], dim=-1)
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
			train_batch (tuple): ({'rgb_static': ..., 'rgb_gripper': ...}, robot_obs)
			batch_idx (int): _description_

		Returns:
			torch.Tensor: _description_
		"""
		x, y = train_batch
		imgs, robot_obs = x

		y_hat = self(imgs, robot_obs)
		loss = F.mse_loss(y_hat, y)
		self.log('train_loss', loss)
		return loss

	# NOTE: Not currently being used
	def test_step(self,
        train_batch: Tuple[Dict, torch.Tensor],
        batch_idx: int,
	) -> torch.Tensor:
		x, y = train_batch
		imgs, robot_obs = x

		y_hat = self(imgs, robot_obs)
		loss = F.mse_loss(y_hat, y)
		self.log('test_loss', loss)
		return loss

	def validation_step(self,
        train_batch: Tuple[Dict, torch.Tensor],
        batch_idx: int,
	) -> torch.Tensor:
		x, y = train_batch
		imgs, robot_obs = x

		y_hat = self(imgs, robot_obs)
		loss = F.mse_loss(y_hat, y)
		self.log('val_loss', loss)
		return loss


