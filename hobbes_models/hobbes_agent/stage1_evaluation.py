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
from stage1_training import get_episode_path, display_frames, preprocess_image
# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"


def initialize_env(robot_obs, scene_obs):
    with initialize(config_path="../../calvin_env/conf/"):
        env_config = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        # env_config["scene"] = "calvin_scene_B"
        env_config.env["use_egl"] = False
        env_config.env["show_gui"] = False
        env_config.env["use_vr"] = False
        env_config.env["use_scene_info"] = True
        env = hydra.utils.instantiate(env_config.env)
    env.reset(robot_obs, scene_obs)
    return env
    
def generate_trajectory(env, model):
    frames = []
    actions = []
    
    for _ in range(100):
        # Get current frame
        curr_frame = env.render(mode="rgb_array")
        
        # Generate action at current frame
        action = model(preprocess_image(curr_frame).unsqueeze(0))
        
        # Save to lists
        frames.append(curr_frame)
        actions.append(action)

        # Force gripper to binary -1 or 1
        action[0][6] = -1 if action[0][6] < 0 else 1

        # Step env
        observation, reward, done, info = env.step(action.detach().squeeze(0))
    
    return frames, actions


    

    
def main():
    # set up the start frame for the episode
    ep = np.load(get_episode_path(360575, HOBBES_DATASET_ROOT_PATH + "/calvin_debug_dataset/training/"))
    
    # initialize env
    env = initialize_env(ep["robot_obs"], ep["scene_obs"])
    
    # load model
    model = Stage1Model.load_from_checkpoint("./checkpoints/task_D_D/turn_on_lightbulb/latest-epoch=999.ckpt")

    # generate sample trajectory
    frames, actions = generate_trajectory(env, model)
    
    display_frames(frames)        

if __name__ == "__main__":
    main()