import os
import sys

# This fixes up dependency issues, with not being able to find dependencies in their expected places
module_paths = []
module_paths.append(os.path.abspath(os.path.join("../../../calvin_env")))
module_paths.append(os.path.abspath(os.path.join("..")))
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from single_task_imitator import SingleTaskImitator
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from hobbes_utils import get_episode_path, preprocess_image, collect_frames
import torchvision
from torchvision import transforms

from transformers import BertTokenizer, BertModel


class BertLanguageEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")        

    def forward(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        return output["pooler_output"]
    

# model = BertLanguageEncoder()

# output = model("hello how are you")
# print(output.squeeze(0))
# print(output.shape)