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
from torch.utils.data import DataLoader
from single_task_imitator import SingleTaskImitator
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from hobbes_utils import *
# NOTE: Run from "hobbes_models/hobbes_agent/"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"

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
            target_task_indxs.append((indx[0], indx[1]+10))
        if (len(target_task_indxs) == num_demonstrations):
            break
    if (len(target_task_indxs) < num_demonstrations):
        print(f"Warning: Requested {num_demonstrations} demonstrations but only found {len(target_task_indxs)}")
    return target_task_indxs

def build_dataloader(dataset_path, task_name, num_demonstrations, batch_size, num_workers, custom_extractor):
    data_path = HOBBES_DATASET_ROOT_PATH + dataset_path
    timeframes = get_task_timeframes(task_name, data_path, num_demonstrations)
    print(f"Found {len(timeframes)} demonstrations for the task {task_name}")
    dataset = []
    for timeframe in timeframes:
        observations, actions = collect_frames(timeframe[0], timeframe[1], data_path, observation_extractor=custom_extractor)
        data_pairs = list(zip(observations, actions))
        dataset.extend(data_pairs)
    print("train dataset len", len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader

def train_model(
    task_name: str = "turn_on_led",
    dataset_path: str = "calvin_debug_dataset",
    model: SingleTaskImitator = None,
    model_save_path: str = "checkpoints/model_params",
    num_demonstrations: int = 1,
    val: bool = False,
    batch_size: int = 16,
    num_workers: int = 1,
    accelerator='cpu',
    max_epochs=1000,
    val_epochs=10, 
    custom_extractor=None
) -> SingleTaskImitator:
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
        model = SingleTaskImitator()
        
    train_dataloader = build_dataloader(dataset_path + '/training/', task_name=task_name, num_demonstrations=num_demonstrations, batch_size=batch_size, num_workers=num_workers, custom_extractor=custom_extractor)
    
    val_dataloader = build_dataloader(dataset_path + '/validation/', task_name=task_name, num_demonstrations=num_demonstrations, batch_size=batch_size, num_workers=num_workers, custom_extractor=custom_extractor) if val else train_dataloader
    
    monitor_str = "val_loss" if val else "train_loss"

    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path, save_top_k=2,
                                          monitor=monitor_str, filename="best-{epoch:02d}-{val_loss:.05f}")
    final_iter_callback = ModelCheckpoint(
        dirpath=model_save_path, save_top_k=2, monitor="epoch", mode="max", filename="latest-{epoch:02d}"
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_save_path)

    trainer = pl.Trainer(
        accelerator=accelerator,
        num_nodes=1,
        precision=16,
        # limit_train_batches=0.5,
        callbacks=[checkpoint_callback, final_iter_callback],
        logger=tb_logger,
        log_every_n_steps=25,
        check_val_every_n_epoch=1,
        max_epochs=max_epochs,
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)

    return model

def main():
    ################ parameters ################
    rgb_static=True
    depth_static=False
    rgb_gripper=False
    depth_gripper=False
    proprioceptive=False
    ################ parameters ################
    
    # Instantiate a new model
    model = SingleTaskImitator(
        rgb_static=rgb_static,
        depth_static=depth_static,
        rgb_gripper=rgb_gripper,
        depth_gripper=depth_gripper,
        proprioceptive=proprioceptive,
    )
    
    # Load a trained model
    # model = SingleTaskImitator.load_from_checkpoint(
    #     "../checkpoints/v2/task_D_D/stack_block/sensors/rgb_static/latest-epoch=99.ckpt",
    #     rgb_static=rgb_static,
    #     depth_static=depth_static,
    #     rgb_gripper=rgb_gripper,
    #     depth_gripper=depth_gripper,
    #     proprioceptive=proprioceptive,
    # )
    
    custom_extractor = get_custom_extractor(
        rgb_static=rgb_static,
        depth_static=depth_static,
        rgb_gripper=rgb_gripper,
        depth_gripper=depth_gripper,
        proprioceptive=proprioceptive,
    )
    
    # train model
    train_model(
        task_name="stack_block",
        dataset_path="task_D_D",
        model=model,
        model_save_path="../checkpoints/v2/task_D_D/stack_block/sensors/rgb_static",
        num_demonstrations=1000,
        batch_size=256,
        num_workers=8,
        accelerator='gpu',
        max_epochs=200,
        val=True,
        val_epochs=1, custom_extractor=custom_extractor)

if __name__ == "__main__":
    main()
