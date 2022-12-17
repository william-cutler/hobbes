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

class VGGVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = torchvision.models.vgg19(pretrained=True)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-7])
        model.classifier = new_classifier
        self.vgg_feature_extractor = model
        
    def forward(self, x):
        x = self.preprocess(x)
        x = self.vgg_feature_extractor(x)
        return x
    
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# frames, actions = collect_frames(
#     start=358483,
#     stop=358484,
#     dataset_path="/home/grail/willaria_research/hobbes/dataset/calvin_debug_dataset/training/",
# )


# img = preprocess(frames[0]).unsqueeze(0)








# model = VGGVisionEncoder()

# output = model(img)
# print(output)
# print(output.shape)