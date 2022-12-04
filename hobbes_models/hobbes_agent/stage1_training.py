import os
import sys

# This fixes up dependency issues, with not being able to find dependencies in their expected places
module_path = os.path.abspath(os.path.join("/home/grail/willaria_research/hobbes/calvin_env"))
if module_path not in sys.path:
    sys.path.append(module_path)

from calvin_env.envs.play_table_env import PlayTableSimEnv
import numpy as np
import torch
from torch.utils.data import DataLoader
from stage1_model import Stage1Model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
import hydra
from hydra import initialize, compose
import cv2

# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"


def get_episode_path(frame_num: int, dataset_path: str, pad_len: int = 7):
    padded_num = str(frame_num).rjust(pad_len, "0")
    return dataset_path + "episode_" + padded_num + ".npz"


def collect_frames(start: int, stop: int, dataset_path: str):
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
        actions.append(torch.from_numpy(np.asarray(ep["rel_actions"])).float())  # 'actions' or 'rel_actions'
    return list(zip(images, actions))


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Prepares a 200 x 200 x 3 RGB image for input into the model. Normalizes to [0, 1] and transposes to [3 x 200 x 200]

    Args:
        img (np.ndarray): Raw scene image.

    Returns:
        torch.Tensor: Transposed image ready to be fed into image encoder.
    """
    return torch.from_numpy(np.transpose(img / 255, (2, 0, 1))).float()


def get_task_timeframes(task_name: str, dataset_path: str) -> list:
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
    return target_task_indxs


def train_model(
    task_name: str = "turn_on_led",
    dataset_path: str = "calvin_debug_dataset",
    model_param_path: str = "checkpoints/model_params",
    val: bool = False,
    batch_size: int = 16,
    num_workers: int = 1,
    num_gpus=0,
) -> Stage1Model:
    """Train a model on a single task.

    Args:
        task_name (str, optional): The name of the single task to train on. Defaults to 'turn_on_led'.
        dataset_path (str, optional): Path to the desired dataset relative to HOBBES_DATASET_ROOT_PATH. Defaults to 'calvin_debug_dataset'.
        model_param_path (str, optional): Path to store model checkpoints in. Defaul\ts to 'checkpoints/model_params'.
        val (bool, optional): Whether to perform validation. Defaults to False.
        batch_size (int, optional): Training batch size. Defaults to 16.
        num_workers (int, optional): Number of workers for training dataloader. Defaults to 1.
        num_gpus (int, optional): Number of GPU's to train on. Defaults to 0 (CPU).
    Returns:
        Stage1Model: The trained model.
    """

    model = Stage1Model()

    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + "/training/"
    train_timeframes = get_task_timeframes(task_name, train_data_path)
    train_dataset = []
    for timeframe in train_timeframes:
        train_dataset.extend(collect_frames(timeframe[0], timeframe[1], train_data_path))
    print("train dataset len", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    val_dataloader = build_val_dataloader(task_name, dataset_path, batch_size, num_workers) if val else train_dataloader
    monitor_str = "val_loss" if val else "train_loss"

    checkpoint_callback = ModelCheckpoint(dirpath=model_param_path, save_top_k=3, monitor=monitor_str)

    final_iter_callback = ModelCheckpoint(
        dirpath=model_param_path, save_top_k=2, monitor="epoch", mode="max", filename="latest-{epoch:02d}"
    )
    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=1,
        precision=16,
        limit_train_batches=0.5,
        callbacks=[checkpoint_callback, final_iter_callback],
        check_val_every_n_epoch=10,
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


def build_frames(model, env_config, num_frames=20):
    """Simulates the trajectory predicted by the model in the environment.

    Args:
        model (_type_): Trained model to predict actions at each timestep.
        env_config (_type_): Config for the environment
        num_frames (int, optional): Number of steps to predict. Defaults to 20.

    Returns:
        _type_: _description_
    """
    env = hydra.utils.instantiate(env_config.env)
    print(env)
    curr_frame = env.render(mode="rgb_array")
    frames = [curr_frame]
    for _ in range(num_frames):
        action = model(preprocess_image(curr_frame).unsqueeze(0))
        print(action)

        # Force gripper to binary -1 or 1
        action[0][6] = -1 if action[0][6] < 0 else 1

        observation, reward, done, info = env.step(action.detach().squeeze(0))
        print(observation['scene_obs'])

        curr_frame = env.render(mode="rgb_array")
        frames.append(curr_frame)

    return frames


# TODO: Figure out how to instantiate environment at particular timestep


def display_frames(frames, title="", ms_per_frame=50):
    """Press any key to start video, once finished press a key to close. DO NOT HIT RED X!

    Args:
        frames (_type_): _description_
        title (str, optional): _description_. Defaults to "".
        ms_per_frame (int, optional): _description_. Defaults to 50.
    """
    for i in range(len(frames)):
        if i == 1:
            _ = cv2.waitKey(0)
        cv2.imshow(title, frames[i][:, :, ::-1])

        key = cv2.waitKey(50)  # pauses for 3 seconds before fetching next image
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_evaluation_env_config(scene_obs_array, robot_obs_array):
    """Config for the evaluation environment copied from RL_WITH_CALVIN.ipynb

    Returns:
        _type_: _description_
    """
    with initialize(config_path="../../calvin_env/conf/"):
        env_config = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        # env_config["scene"] = "calvin_scene_B"
        env_config.env["use_egl"] = False
        env_config.env["show_gui"] = False
        env_config.env["use_vr"] = False
        env_config.env["use_scene_info"] = True
        
        set_scene_config(env_config.scene, scene_obs_array)
        set_robot_config(env_config.robot, robot_obs_array)
        
        print(env_config.scene.objects.fixed_objects.table.joints.base__slide.initial_state)
        

    return env_config

def set_scene_config(scene_config, scene_obs_array):
    """(dtype=np.float32, shape=(24,))
        sliding door (1): joint state
        drawer (1): joint state
        button (1): joint state
        switch (1): joint state
        lightbulb (1): on=1, off=0
        green light (1): on=1, off=0
        red block (6): (x, y, z, euler_x, euler_y, euler_z)
        blue block (6): (x, y, z, euler_x, euler_y, euler_z)
        pink block (6): (x, y, z, euler_x, euler_y, euler_z)

    Args:
        scene_config (_type_): _description_
        scene_obs_array (_type_): _description_
    """
    table = scene_config.objects.fixed_objects.table
    print(type(table.joints.base__slide.initial_state))
    print(table.joints.base__slide.initial_state)
    print(scene_obs_array)
    table.joints.base__slide.initial_state = 0.30 #float(scene_obs_array[0])
    table.joints.base__drawer.initial_state = float(scene_obs_array[1])

    table.buttons.base__button.initial_state = float(scene_obs_array[2])
    table.switches.base__switch.initial_state = 0.1 # float(scene_obs_array[3])
    
    # TODO: Warning we can't set light state directly for first frame
    
    blocks = scene_config.objects.movable_objects
    
    def set_block(block_name:str, scene_obs_start:int):
        blocks[block_name]["initial_pos"] = [float(x) for x in list(scene_obs_array[scene_obs_start:scene_obs_start + 3])]
        blocks[block_name]["initial_orn"] = [float(x) for x in list(scene_obs_array[scene_obs_start+3:scene_obs_start + 6])]

    set_block("block_red", 6)
    set_block("block_blue", 12)
    set_block("block_pink", 18)


def set_robot_config(robot_config, robot_obs):
    pass

def main():
    # train model
    model = train_model(
        ask_name="turn_on_lightbulb",
        dataset_path="task_D_D",
        model_param_path="checkpoints/task_D_D/turn_on_lightbulb",
        batch_size=128,
        num_workers=1,
        num_gpus=1)

    # # load model
    # ep = np.load(get_episode_path(360575, HOBBES_DATASET_ROOT_PATH + "/calvin_debug_dataset/training/"))
    # env_config = create_evaluation_env_config(ep["scene_obs"], ep["robot_obs"])

    # model = Stage1Model.load_from_checkpoint("./checkpoints/task_D_D/turn_on_lightbulb/latest-epoch=999.ckpt")
    # frames = build_frames(model, env_config, num_frames=100)
    # display_frames(frames)
    # #360575

if __name__ == "__main__":
    main()
