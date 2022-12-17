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
from multi_task_model import MultiTaskModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from hobbes_utils import get_episode_path, preprocess_image
from bert_language_encoder import BertLanguageEncoder
from vgg_vision_encoder import VGGVisionEncoder

import torchvision
from torchvision import transforms

# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"

def get_task_timeframes_with_language(task_set, dataset_path: str, num_demonstrations: int, buffer: int = 10) -> list:
    """Returns the start and end of every episode corresponding to the specified task.

    Args:
        task_name (str): Name of the task to extract.
        dataset_path (str): Path to the dataset

    Returns:
        list: List of pairs (episode_start_index, episode_end_index)
    """
    lang = np.load(dataset_path + "lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    lang = dict(enumerate(lang.flatten(), 1))[1]

    target_task_infos = []
    for (annotation, task, indx) in zip(lang["language"]["ann"], lang["language"]["task"], lang["info"]["indx"]):
        if task in task_set:
            target_task_infos.append({"start": indx[0], "stop": indx[1] + buffer, "task": task, "annotation": annotation})
        if (len(target_task_infos) == num_demonstrations):
            break
    if (len(target_task_infos) < num_demonstrations):
        print(f"Warning: Requested {num_demonstrations} demonstrations but only found {len(target_task_infos)}")
    return target_task_infos


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: MultiTaskModel = None,
    model_save_path: str = "checkpoints/model_params",
    val: bool = False,
    accelerator='cpu',
    max_epochs=1000,
) -> MultiTaskModel:
    """Train a model on a single task.

    Args:
        task_set (str, optional): The name of the single task to train on. Defaults to 'turn_on_led'.
        dataset_path (str, optional): Path to the desired dataset relative to HOBBES_DATASET_ROOT_PATH. Defaults to 'calvin_debug_dataset'.
        model_save_path (str, optional): Path to store model checkpoints in. Defaul\ts to 'checkpoints/model_params'.
        val (bool, optional): Whether to perform validation. Defaults to False.
        batch_size (int, optional): Training batch size. Defaults to 16.
        num_workers (int, optional): Number of workers for training dataloader. Defaults to 1.
        num_gpus (int, optional): Number of GPU's to train on. Defaults to 0 (CPU).
    Returns:
        Stage1ModelV2: The trained model.
    """

    if not model:
        model = MultiTaskModel()


    
    monitor_str = "val_loss" if val else "train_loss"
    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path, save_top_k=2,
                                          monitor=monitor_str, filename="best-{epoch:02d}-{val_loss:.05f}")
    final_iter_callback = ModelCheckpoint(
        dirpath=model_save_path, save_top_k=2, monitor="epoch", mode="max", filename="latest-{epoch:02d}"
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/model3/")

    trainer = pl.Trainer(
        accelerator=accelerator,
        num_nodes=1,
        precision=16,
        # limit_train_batches=0.5,
        callbacks=[checkpoint_callback, final_iter_callback],
        logger=tb_logger,
        log_every_n_steps=25,
        check_val_every_n_epoch=10,
        max_epochs=max_epochs,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


def collect_frames(start: int, stop: int, dataset_path: str, lang_annotation, action_type: str = "rel_actions"):
    """Loads the specified range of episodes (semi-closed) for creating a demonstration video to imitate.

    Args:
        start (_type_): _description_
        stop (_type_): _description_

    Returns:
        _type_: List of pairs of static camera images and action taken at that step.
    """
    observations = []
    actions = []
    for i in range(start, stop):
        ep = np.load(get_episode_path(i, dataset_path))
        imgs = {
            'rgb_static': preprocess_image(ep["rgb_static"]),
            'rgb_gripper': preprocess_image(ep["rgb_gripper"])
        }
        robot_obs = torch.tensor(ep['robot_obs']).float()
        
        observations.append((imgs, robot_obs, lang_annotation))
        actions.append(torch.tensor(ep[action_type]).float())  # 'actions' or 'rel_actions'
    return list(zip(observations, actions))

# ve = VGGVisionEncoder()

# def preprocess_image(img: np.ndarray) -> torch.Tensor:
#     """Prepares a 200 x 200 x 3 RGB image for input into the model. Normalizes to [0, 1] and transposes to [3 x 200 x 200]

#     Args:
#         img (np.ndarray): Raw scene image.

#     Returns:
#         torch.Tensor: Transposed image ready to be fed into image encoder.
#     """
#     preprocess = torch.from_numpy(np.transpose(img / 255, (2, 0, 1))).float()
#     encoding = ve(preprocess)
#     return encoding


def build_finetune_data_loader(data_path, task_name, num_demonstrations, batch_size, num_workers):
    language_encoder = BertLanguageEncoder()
    timeframes = get_task_timeframes_with_language(task_name, data_path, num_demonstrations)
    print("Building dataset from ", data_path)
    print(f"Found {len(timeframes)} demonstrations for the task {task_name}")
    dataset = []
    for task_info in timeframes:
        dataset.extend(collect_frames(task_info["start"], task_info["stop"], data_path, language_encoder(task_info["annotation"]).squeeze(0).detach())) # Now with the language annotation text!

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

def build_pretrain_data_loader(data_path, language_embedding, batch_size, num_workers):

    print("Building dataset from ", data_path)
    ep_range_idx = np.load(data_path + "/ep_start_end_ids.npy")[0]
    dataset = []
    dataset.extend(collect_frames(ep_range_idx[0], ep_range_idx[1] + 1, data_path, language_embedding.squeeze(0).detach())) # Now with the language annotation text!
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def main_finetune():
    ################ parameters ################
    model = MultiTaskModel()
    MultiTaskModel.load_from_checkpoint("../checkpoints/v3/task_D_D/pretrain-epoch=999.ckpt")
    dataset_path="task_D_D"
    task_set={"turn_off_lightbulb", "turn_on_lightbulb", "turn_off_led", "turn_on_led"}
    num_demonstrations=1000
    batch_size=256
    num_workers=8
    val=True
    ################ parameters ################
    
    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + "/training/"
    val_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + "/validation/"

    train_dataloader = build_finetune_data_loader(
        train_data_path, task_set, num_demonstrations=num_demonstrations, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = build_finetune_data_loader(
        val_data_path, task_set, num_demonstrations=num_demonstrations, batch_size=batch_size, num_workers=num_workers) if val else train_dataloader
    
    train_model(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        model_save_path="../checkpoints/v3/task_D_D/finetune/lights",
        val=val,
        accelerator='gpu',
        max_epochs=1000
    )


def main_pretrain():
    ################ parameters ################
    model = None
    dataset_path="task_D_D"
    batch_size=256
    num_workers=8
    val=True
    ################ parameters ################    
    
    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + "/training/"
    val_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + "/validation/"

    PLAY_DATA_LANGUAGE_TOKEN = "unspecified task"
    language_encoder = BertLanguageEncoder()
    play_token_encoding = language_encoder(PLAY_DATA_LANGUAGE_TOKEN)
    train_dataloader = build_pretrain_data_loader(train_data_path, language_embedding=play_token_encoding, batch_size=batch_size, num_workers=num_workers)    
    val_dataloader = build_pretrain_data_loader(val_data_path, language_embedding=play_token_encoding, batch_size=batch_size, num_workers=num_workers) if val else train_dataloader
    
    train_model(train_dataloader=train_dataloader, val_dataloader=val_dataloader, model = model, model_save_path="../checkpoints/v3/task_D_D", val=val, accelerator='gpu', max_epochs=1000)

if __name__ == "__main__":
    ################ parameters ################
    finetune = False
    ################ parameters ################
    
    if finetune:
        main_finetune()
    else:
        main_pretrain()