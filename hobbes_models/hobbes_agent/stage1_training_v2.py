import os
import sys

# This fixes up dependency issues, with not being able to find dependencies in their expected places
module_path = os.path.abspath(os.path.join("/home/grail/willaria_research/hobbes/calvin_env"))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
from torch.utils.data import DataLoader
from stage1_model import Stage1ModelV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from stage1_utils import get_episode_path, preprocess_image, save_gif

# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"

def collect_frames(start: int, stop: int, dataset_path: str, action_type: str = "rel_actions"):
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
        
        observations.append((imgs, robot_obs))
        actions.append(torch.tensor(ep[action_type]).float())  # 'actions' or 'rel_actions'
    return list(zip(observations, actions))



def get_task_timeframes(task_name: str, dataset_path: str, num_demonstrations: int) -> list:
    """Returns the start and end of every episode corresponding to the specified task.

    Args:
        task_name (str): Name of the task to extract.
        dataset_path (str): Path to the dataset

    Returns:
        list: List of pairs (episode_start_index, episode_end_index)
    """
    lang = np.load(dataset_path + "lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    lang = dict(enumerate(lang.flatten(), 1))[1]

    target_task_indxs = []
    for (task, indx) in zip(lang["language"]["task"], lang["info"]["indx"]):
        if task == task_name:
            target_task_indxs.append((indx[0]-10, indx[1]+50))
        if (len(target_task_indxs) == num_demonstrations):
            break
    if (len(target_task_indxs) < num_demonstrations):
        print(f"Warning: Requested {num_demonstrations} demonstrations but only found {len(target_task_indxs)}")
    return target_task_indxs


def train_model(
    task_name: str = "turn_on_led",
    dataset_path: str = "calvin_debug_dataset",
    model: Stage1ModelV2 = None,
    model_save_path: str = "checkpoints/model_params",
    num_demonstrations: int = 1,
    val: bool = False,
    batch_size: int = 16,
    num_workers: int = 1,
    num_gpus=0,
    max_epochs=1000,
    val_epochs=10
) -> Stage1ModelV2:
    """Train a model on a single task.

    Args:
        task_name (str, optional): The name of the single task to train on. Defaults to 'turn_on_led'.
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
        model = Stage1ModelV2()

    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + "/training/"
    train_timeframes = get_task_timeframes(task_name, train_data_path, num_demonstrations)
    print(f"Found {len(train_timeframes)} demonstrations for the task {task_name}")
    train_dataset = []
    for timeframe in train_timeframes:
        train_dataset.extend(collect_frames(timeframe[0], timeframe[1], train_data_path))
    print("train dataset len", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    val_dataloader = build_val_dataloader(task_name, dataset_path, batch_size, num_workers) if val else train_dataloader
    monitor_str = "val_loss" if val else "train_loss"

    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path, save_top_k=3, monitor=monitor_str)
    final_iter_callback = ModelCheckpoint(
        dirpath=model_save_path, save_top_k=2, monitor="epoch", mode="max", filename="latest-{epoch:02d}"
    )

    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=1,
        precision=16,
        limit_train_batches=0.5,
        callbacks=[checkpoint_callback, final_iter_callback],
        check_val_every_n_epoch=val_epochs,
        max_epochs=max_epochs
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


def build_val_dataloader(task_name: str, dataset_path: str, batch_size: int = 16, num_workers: int = 1) -> DataLoader:
    """Builds the

    Args:
        task_name (str): _description_
        dataset_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 16.
        num_workers (int, optional): _description_. Defaults to 1.

    Returns:
        DataLoader: _description_
    """
    val_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + "/validation/"
    val_timeframes = get_task_timeframes(task_name, val_data_path)
    val_dataset = []
    for timeframe in val_timeframes:
        val_dataset.extend(collect_frames(timeframe[0], timeframe[1], val_data_path))
    print("val dataset len", len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    return val_dataloader

def main():
    # model = Stage1ModelV2.load_from_checkpoint("./checkpoints/calvin_debug_dataset/turn_off_lightbulb/latest-epoch=999-v2.ckpt")
    model = None

    # train model
    train_model(
        task_name="turn_off_lightbulb",
        dataset_path="calvin_debug_dataset",
        model=model,
        model_save_path="checkpoints/v2/calvin_debug_dataset/turn_off_lightbulb/1000",
        batch_size=16,
        num_workers=1,
        num_gpus=1,
        max_epochs=1000,
        val_epochs=10)

    # # load model
    # ep = np.load(get_episode_path(360575, HOBBES_DATASET_ROOT_PATH + "/calvin_debug_dataset/training/"))
    # env_config = create_evaluation_env_config(ep["scene_obs"], ep["robot_obs"])

    # model = Stage1ModelV2.load_from_checkpoint("./checkpoints/task_D_D/turn_on_lightbulb/latest-epoch=999.ckpt")
    # frames = build_frames(model, env_config, num_frames=100)
    # display_frames(frames)
    # #360575

if __name__ == "__main__":
    main()
