import os
import sys

# This fixes up dependency issues, with not being able to find dependencies in their expected places
module_paths = []
module_paths.append(os.path.abspath(os.path.join("../../../calvin_env")))
module_paths.append(os.path.abspath(os.path.join("..")))
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

from hobbes_utils import get_episode_path, preprocess_image
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from model1 import Model1
from torch.utils.data import DataLoader
import torch
import numpy as np



def collect_frames(start: int, stop: int, dataset_path: str, action_type: str):
    """Loads the specified range of episodes (semi-closed) for creating a demonstration video to imitate.

    Args:
        start (_type_): _description_
        stop (_type_): _description_

    Returns:
        _type_: List of pairs of static camera images and action taken at that step.
    """
    images = []
    actions = []
    for i in range(start, stop):
        ep = np.load(get_episode_path(i, dataset_path))
        images.append(preprocess_image(ep["rgb_static"]))
        actions.append(torch.from_numpy(np.asarray(ep[action_type])).float())  # 'actions' or 'rel_actions'
    #save_gif(images, "target.gif")
    return list(zip(images, actions))


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
            target_task_indxs.append(indx)
        if len(target_task_indxs) == num_demonstrations:
            break
    if len(target_task_indxs) < num_demonstrations:
        print(f"Warning: Requested {num_demonstrations} demonstrations but only found {len(target_task_indxs)}")
    return target_task_indxs


def train_model(cfg, model: Model1 = None) -> Model1:
    """Train a model on a single task.

    Args:

    Returns:
        Stage1Model: The trained model.
    """

    if not model:
        model = Model1()

    train_data_path = cfg.absolute_dataset_path + "/training/"
    train_timeframes = get_task_timeframes(cfg.task_name, train_data_path, cfg.num_demonstrations)
    print(f"Found {len(train_timeframes)} demonstrations for the task {cfg.task_name}")
    train_dataset = []
    for timeframe in train_timeframes:
        train_dataset.extend(collect_frames(timeframe[0], timeframe[1], train_data_path, cfg.action_type))
    print("train dataset len", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    val_dataloader = build_val_dataloader(val_timeframes=train_timeframes) if cfg.val else train_dataloader
    monitor_str = "val_loss" if cfg.val else "train_loss"

    checkpoint_callback = ModelCheckpoint(dirpath=cfg.model_save_path, save_top_k=2,
                                          monitor=monitor_str, filename="best-{train_loss:.05f}")
    final_iter_callback = ModelCheckpoint(
        dirpath=cfg.model_save_path, save_top_k=2, monitor="epoch", mode="max", filename="latest-{epoch:02d}"
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/model1/")

    trainer = pl.Trainer(
        gpus=cfg.num_gpus,
        num_nodes=1,
        precision=16,
        # limit_train_batches=0.5,
        callbacks=[checkpoint_callback, final_iter_callback],
        logger=tb_logger,
        check_val_every_n_epoch=cfg.epochs_between_val,
        max_epochs=cfg.max_epochs,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


def build_val_dataloader(cfg, val_timeframes) -> DataLoader:
    val_data_path = cfg.absolute_dataset_path + "/validation/"
    val_dataset = []
    for timeframe in val_timeframes:
        val_dataset.extend(collect_frames(timeframe[0], timeframe[1], val_data_path, cfg.action_type))
    print("val dataset len", len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return val_dataloader


@hydra.main(config_path="conf", config_name="stage1_model_config")
def main(cfg):

    # train model
    train_model(cfg, model=Model1())

    hydra.core.global_hydra.GlobalHydra.instance().clear()


# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
if __name__ == "__main__":
    main()
