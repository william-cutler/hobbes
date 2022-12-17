import os
import sys

# This fixes up dependency issues, with not being able to find dependencies in their expected places
module_path = os.path.abspath(os.path.join("/home/grail/willaria_research/hobbes/calvin_env"))
if module_path not in sys.path:
    sys.path.append(module_path)

from hobbes_utils import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from lstm2_model import HobbesDecoderWrapper
from lstm2_dataset import LSTMDataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np



# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"


def train_model(
    task_name: str = "turn_on_led",
    dataset_path: str = "calvin_debug_dataset",
    model: HobbesDecoderWrapper = None,
    model_save_path: str = "checkpoints/model_params",
    num_demonstrations: int = 1,
    val: bool = False,
    batch_size: int = 16,
    num_workers: int = 1,
    num_gpus=0,
    max_epochs=1000,
    val_epochs=10
) -> HobbesDecoderWrapper:
    """Train a model on a single task.

    Args:
        task_name (str, optional): The name of the single task to train on. Defaults to 'turn_on_led'.
        dataset_path (str, optional): Path to the desired dataset relative to HOBBES_DATASET_ROOT_PATH. Defaults to 'calvin_debug_dataset'.
        model_save_path (str, optional): Path to store model checkpoints in. Defaults to 'checkpoints/model_params'.
        val (bool, optional): Whether to perform validation. Defaults to False.
        batch_size (int, optional): Training batch size. Defaults to 16.
        num_workers (int, optional): Number of workers for training dataloader. Defaults to 1.
        num_gpus (int, optional): Number of GPU's to train on. Defaults to 0 (CPU).
    Returns:
        Stage1ModelV2: The trained model.
    """

    if not model:
        model = HobbesDecoderWrapper()

    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + "/training/"

    train_dataset = LSTMDataset(task_name, train_data_path, num_demonstrations)

    print("train dataset len", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    val_dataloader = build_val_dataloader(task_name, dataset_path, batch_size, num_workers) if val else train_dataloader
    monitor_str = "val_loss" if val else "train_loss"

    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path, save_top_k=3, monitor=monitor_str)
    final_iter_callback = ModelCheckpoint(
        dirpath=model_save_path, save_top_k=2, monitor="epoch", mode="max", filename="latest-{epoch:02d}"
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/lstm/debug")

    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=1,
        precision=16,
        # limit_train_batches=0.5,
        callbacks=[checkpoint_callback, final_iter_callback],
        logger=tb_logger,
        check_val_every_n_epoch=1,
        max_epochs=max_epochs,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


def build_val_dataloader(task_name: str, dataset_path: str, batch_size: int = 16, num_demonstrations=1, num_workers: int = 1) -> DataLoader:
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

    val_dataset = LSTMDataset(task_name, val_data_path, num_demonstrations)

    print("val dataset len", len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    return val_dataloader


def main():
    # model = Stage1ModelV2.load_from_checkpoint("./checkpoints/calvin_debug_dataset/turn_off_lightbulb/latest-epoch=999-v2.ckpt")
    model = None

    # train model
    train_model(
        task_name="turn_off_lightbulb",
        dataset_path="task_D_D",
        model=model,
        model_save_path="checkpoints/lstm2/task_D_D/turn_off_lightbulb",
        num_demonstrations=1000,
        val=True,
        batch_size=16,
        num_workers=8,
        num_gpus=1,
        max_epochs=1000,
        val_epochs=1)

    # # load model
    # ep = np.load(get_episode_path(360575, HOBBES_DATASET_ROOT_PATH + "/calvin_debug_dataset/training/"))
    # env_config = create_evaluation_env_config(ep["scene_obs"], ep["robot_obs"])

    # model = Stage1ModelV2.load_from_checkpoint("./checkpoints/task_D_D/turn_on_lightbulb/latest-epoch=999.ckpt")
    # frames = build_frames(model, env_config, num_frames=100)
    # display_frames(frames)
    # #360575


if __name__ == "__main__":
    main()
