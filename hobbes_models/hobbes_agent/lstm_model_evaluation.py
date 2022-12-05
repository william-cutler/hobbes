from lstm_model_training import LSTMDataset, collect_frames, get_task_timeframes
from lstm_model import HobbesLSTM
from hobbes_utils import *
import cv2
import hydra
import torch
import numpy as np
import os
import sys

# This fixes up dependency issues, with not being able to find dependencies in their expected places
module_path = os.path.abspath(os.path.join("/home/grail/willaria_research/hobbes/calvin_env"))
if module_path not in sys.path:
    sys.path.append(module_path)


# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"


def initialize_env(robot_obs, scene_obs):
    with hydra.initialize(config_path="../../calvin_env/conf/"):
        env_config = hydra.compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        # env_config["scene"] = "calvin_scene_B"
        # env_config.env["use_egl"] = False
        # env_config.env["show_gui"] = False
        env_config.env["use_vr"] = False
        # env_config.env["use_scene_info"] = True
        env = hydra.utils.instantiate(env_config.env)
    env.reset(robot_obs, scene_obs)
    return env


def generate_trajectory(env, model, num_frames=20):
    """Simulates the trajectory predicted by the model in the environment.

    Args:
        model (_type_): Trained model to predict actions at each timestep.
        env_config (_type_): Config for the environment
        num_frames (int, optional): Number of steps to predict. Defaults to 20.

    Returns:
        _type_: _description_
    """
    frames = []
    actions = []

    for i in range(num_frames):
        # Get current frame
        curr_frame = env.render(mode="rgb_array")

        obs = env.get_obs()
        # unsqueeze to emulate batch size of 1
        curr_image = preprocess_image(obs["rgb_obs"]["rgb_static"]).unsqueeze(0).float()
        # Generate action at current frame
        train_timeframes = get_task_timeframes(
            target_task_name="turn_off_lightbulb", dataset_path=HOBBES_DATASET_ROOT_PATH + "calvin_debug_dataset/training/", num_demonstrations=1)
        observations, actions = collect_frames(train_timeframes[0][0], train_timeframes[0][1], 
                                                observation_extractor=static_image_extractor, 
                                                dataset_path=HOBBES_DATASET_ROOT_PATH + "calvin_debug_dataset/training/")

        action = model(torch.stack(observations).unsqueeze(0).float(), torch.tensor([len(observations)]).float(), curr_image)

        # Save to lists
        frames.append(curr_frame)
        actions.append(action)

        # Force gripper to binary -1 or 1
        action[0][6] = -1 if action[0][6] < 0 else 1

        action_dict = {"type": "cartesian_rel", "action": action.detach().squeeze(0)}
        # Step env
        observation, reward, done, info = env.step(action_dict)

        if i % 10 == 0:
            print("Processed frame", i)

    return frames, actions


def main():
    # set up the start frame for the episode
    ep = np.load(get_episode_path(360571, HOBBES_DATASET_ROOT_PATH + "/calvin_debug_dataset/training/"))

    # initialize env
    env = initialize_env(ep["robot_obs"], ep["scene_obs"])

    # load model
    #model = Stage1Model.load_from_checkpoint("./checkpoints/task_D_D/turn_on_lightbulb/latest-epoch=999.ckpt")
    model = HobbesLSTM.load_from_checkpoint(
        "./checkpoints/lstm/calvin_debug_dataset/turn_off_lightbulb/latest-epoch=999.ckpt")
    
    # generate sample trajectory
    frames, actions = generate_trajectory(env, model, num_frames=400)

    save_gif(frames, file_name='lstm.gif')
    # display_frames(frames)


if __name__ == "__main__":
    main()
