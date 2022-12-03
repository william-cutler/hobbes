from pathlib import Path
import sys

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
print(Path(__file__).absolute().parents[1])

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
HOBBES_DATASET_ROOT_PATH = '/home/grail/willaria_research/hobbes/dataset/'

def get_episode_path(frame_num, dataset_path, pad_len=7):
    padded_num = str(frame_num).rjust(pad_len, '0')
    return dataset_path + 'episode_' + padded_num + '.npz'

def collect_frames(start, stop, dataset_path):
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
        actions.append(torch.from_numpy(np.asarray(ep["rel_actions"])).float()) # 'actions' or 'rel_actions'
    return list(zip(images, actions))

def preprocess_image(img):
    """Prepares a 200 x 200 x 3 RGB image for input into the model. Normalizes to [0, 1] and transposes to [3 x 200 x 200]

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.from_numpy(np.transpose(img / 255, (2, 0, 1))).float()

def get_task_timeframes(task_name, dataset_path):
    """Returns the start and end of every episode corresponding to the specified task.

    Args:
        task_name (_type_): Name of the task to extract.
        dataset_path (_type_): Path to the dataset 

    Returns:
        _type_: List of pairs (episode_start_index, episode_end_index)
    """
    lang = np.load(dataset_path + "lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    lang = dict(enumerate(lang.flatten(), 1))[1]

    target_task_indxs = []
    for (task, indx) in zip(lang['language']['task'], lang['info']['indx']):
        if task == task_name:
            target_task_indxs.append(indx)
    return target_task_indxs

def train_model(task_name ='turn_on_led', dataset_path='debug_dataset', val=True):
    """Train a model on a single task.

    Args:
        task_name (str, optional): The name of the single task to train on. Defaults to 'turn_on_led'.
        dataset_path (str, optional): Path to the desired dataset relative to HOBBES_DATASET_ROOT_PATH. Defaults to 'debug_dataset'.
        val (bool, optional): Whether to perform validation. Defaults to True.

    Returns:
        _type_: The trained model.
    """
    model = Stage1Model()

    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + '/training/'
    train_timeframes = get_task_timeframes(task_name, train_data_path)
    train_dataset = []
    for timeframe in train_timeframes:
        train_dataset.extend(collect_frames(timeframe[0], timeframe[1], train_data_path))
    print('train dataset len', len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=24)

    val_dataloader = build_val_dataloader(task_name, dataset_path) if val else None

    checkpoint_callback = ModelCheckpoint(dirpath="model_checkpoints_rel_1/", save_top_k=3, monitor="val_loss")
    trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=0.5, callbacks=[checkpoint_callback], check_val_every_n_epoch=10, default_root_dir='./model_params_2')
    trainer.fit(model, train_dataloader, val_dataloader)

    return model

def build_val_dataloader(task_name, dataset_path):
    val_data_path = HOBBES_DATASET_ROOT_PATH + dataset_path + '/validation/'
    val_timeframes = get_task_timeframes(task_name, val_data_path)
    val_dataset = []
    for timeframe in val_timeframes:
        val_dataset.extend(collect_frames(timeframe[0], timeframe[1], val_data_path))
    print('val dataset len', len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=128, num_workers=24)
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
    curr_frame = env.render(mode="rgb_array")
    frames = [curr_frame]
    for _ in range(num_frames):
        action = model(preprocess_image(curr_frame).unsqueeze(0))
        print(action)

        # Force gripper to binary -1 or 1
        action[0][6] = -1 if action[0][6] < 0 else 1

        observation, reward, done, info = env.step(action.detach().squeeze(0))
        
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
        cv2.imshow(title, frames[i][:,:,::-1])
        
        key = cv2.waitKey(50)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_evaluation_env_config():
    """Config for the evaluation environment copied from RL_WITH_CALVIN.ipynb

    Returns:
        _type_: _description_
    """
    with initialize(config_path="../../calvin_env/conf/"):
        env_config = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        #env_config["scene"] = "calvin_scene_B"
        env_config.env["use_egl"] = False
        env_config.env["show_gui"] = False
        env_config.env["use_vr"] = False
        env_config.env["use_scene_info"] = True

    return env_config

def main():    
    # train model
    # model = train_model(dataset_path='task_D_D')

    # load model
    model = Stage1Model.load_from_checkpoint("./model_checkpoints_abs_1/epoch=529-step=18019.ckpt")
    
    env_config = create_evaluation_env_config()
    frames = build_frames(model, env_config, num_frames=200)
    display_frames(frames)

if __name__ == "__main__":
    main()