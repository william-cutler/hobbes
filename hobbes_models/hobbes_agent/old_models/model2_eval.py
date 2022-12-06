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
from model2 import Model2
import hydra
from torch import nn
import torch
import cv2
from model2_training import collect_frames, get_task_timeframes
from hobbes_utils import preprocess_image, get_episode_path, save_gif


# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"


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
        imgs = {
            'rgb_static': preprocess_image(obs["rgb_obs"]["rgb_static"]).unsqueeze(0),
            'rgb_gripper': preprocess_image(obs["rgb_obs"]["rgb_gripper"]).unsqueeze(0)
        }
        robot_obs = torch.tensor(obs['robot_obs']).float().unsqueeze(0)
        # Generate action at current frame
        action = model(imgs, robot_obs)
        
        # Save to lists
        frames.append(curr_frame)
        actions.append(action)

        # Force gripper to binary -1 or 1
        action[0][6] = -1 if action[0][6] < 0 else 1

        action_dict = {"type":"cartesian_rel", "action":action.detach().squeeze(0)}
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

        key = cv2.waitKey(ms_per_frame)  # pauses for 50ms before fetching next image
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def main():
    dataset_name = 'task_D_D'
    task_name = "stack_block"


    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_name + '/training/'
    val_data_path = HOBBES_DATASET_ROOT_PATH + dataset_name + '/validation/'

    for train_or_val in ('train', 'val'):
        if train_or_val == 'train':
            data_path = train_data_path 
            #continue
        elif train_or_val == 'val':
            data_path = val_data_path
        print('using dataset', data_path)

        # set up the start frame for the episode
        episode_ranges = get_task_timeframes(task_name, 
                                            data_path,
                                            1000)

        for episode_range in episode_ranges:
            print(f"Using episode in the range(s): {episode_range} for task {task_name} in {train_or_val} dataset {dataset_name}")

            ep = np.load(
                get_episode_path(episode_range[0], data_path)
            )

            data = collect_frames(episode_range[0], episode_range[1], data_path)
            target_actions = [y for x, y in data]

            # initialize env
            env = initialize_env(ep["robot_obs"], ep["scene_obs"])

            # load model
            model = Model2.load_from_checkpoint(
                "../checkpoints/v2/task_D_D/" + task_name + "/latest-epoch=999.ckpt"
            )

            # generate sample trajectory
            desired_num_frames = len(target_actions)
            frames, actions = generate_trajectory(env, model, desired_num_frames)

            save_gif(frames, file_name=f"model2-{train_or_val}-({task_name})-({episode_range[0]}-{episode_range[1]}).gif")

            loss = nn.MSELoss(reduction='mean')

            target_actions = torch.stack(target_actions[:desired_num_frames])
            actions = torch.stack(actions).squeeze(1)

            print(f"Eval loss for trajectory of {len(frames)} timesteps: {loss(input=actions, target=target_actions)}")

if __name__ == "__main__":
    main()