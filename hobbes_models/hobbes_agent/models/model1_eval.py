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
from model1 import Model1
import hydra
from torch import nn
import torch
import cv2
from hobbes_utils import *

HOBBES_TRAIN_DATA_PATH =  "/home/grail/willaria_research/hobbes/dataset/calvin_debug_dataset/training/"

def initialize_env(robot_obs, scene_obs):
    with hydra.initialize(config_path="../../../calvin_env/conf/"):
        env_config = hydra.compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        # env_config["scene"] = "calvin_scene_B"
        # env_config.env["use_egl"] = False
        # env_config.env["show_gui"] = False
        env_config.env["use_vr"] = False
        # env_config.env["use_scene_info"] = True
        env = hydra.utils.instantiate(env_config.env)
    env.reset(robot_obs, scene_obs)
    return env


def generate_trajectory(num_frames, env, model):
    """Simulates the trajectory predicted by the model in the environment.

    Args:

    Returns:
        _type_: _description_
    """
    frames = []
    actions = []

    for i in range(num_frames):
        # Get current frame
        curr_frame = env.render(mode="rgb_array")

        # Generate action at current frame
        action = model(preprocess_image(curr_frame).unsqueeze(0))

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

        key = cv2.waitKey(ms_per_frame)
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#@hydra.main(config_path="conf", config_name="stage1_eval_config")
def main():
    #hydra.core.global_hydra.GlobalHydra.instance().clear()
    # set up the start frame for the episode
    task_name = "turn_off_lightbulb"
    episode_range = get_task_timeframes(task_name, 
                                        HOBBES_TRAIN_DATA_PATH,
                                        1)[0]
                                        
    print(f"Using episode in the range(s): {episode_range} for task {task_name}")

    ep = np.load(
        get_episode_path(episode_range[0], HOBBES_TRAIN_DATA_PATH)
    )

    target_obs, target_actions = collect_frames(episode_range[0], episode_range[1], HOBBES_TRAIN_DATA_PATH)


    # initialize env
    env = initialize_env(ep["robot_obs"], ep["scene_obs"])

    # load model
    # model = Stage1Model.load_from_checkpoint("./checkpoints/task_D_D/turn_on_lightbulb/latest-epoch=999.ckpt")
    model = Model1.load_from_checkpoint(
        "../checkpoints/v1/calvin_debug_dataset/turn_off_lightbulb/latest-epoch=999.ckpt"
    )

    # generate sample trajectory
    desired_num_frames = 3 * len(target_actions)
    frames, actions = generate_trajectory(desired_num_frames, env, model)

    save_gif(frames, file_name="model1-predicted.gif")
    save_gif(target_obs, file_name="model1-target.gif")

    loss = nn.MSELoss(reduction='mean')

    target_actions = torch.stack(target_actions)
    actions = torch.stack(actions[:len(target_actions)]).squeeze(1)

    print(f"Eval loss for trajectory of {len(frames)} timesteps: {loss(input=actions, target=target_actions)}")
    # display_frames(frames)
    #hydra.core.global_hydra.GlobalHydra.instance().clear()


# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
if __name__ == "__main__":
    main()
