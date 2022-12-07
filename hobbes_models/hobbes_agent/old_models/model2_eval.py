import os
import sys

# This fixes up dependency issues, with not being able to find dependencies in their expected places
module_paths = []
module_paths.append(os.path.abspath(os.path.join("../../../calvin_env")))
module_paths.append(os.path.abspath(os.path.join("..")))
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

import csv
import numpy as np
from model2 import Model2
import hydra
from torch import nn
import torch
from hobbes_utils import *
from typing import List
from termcolor import colored


# NOTE: Run from "hobbes_models/hobbes_agent/", "conda activate calvin_conda_env"
HOBBES_DATASET_ROOT_PATH = "/home/grail/willaria_research/hobbes/dataset/"


# def initialize_env(robot_obs, scene_obs):
#     with hydra.initialize(config_path="../../../calvin_env/conf/"):
#         env_config = hydra.compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
#         # env_config["scene"] = "calvin_scene_B"
#         # env_config.env["use_egl"] = False
#         # env_config.env["show_gui"] = False
#         env_config.env["use_vr"] = False
#         # env_config.env["use_scene_info"] = True
#         env = hydra.utils.instantiate(env_config.env)
#     env.reset(robot_obs, scene_obs)
#     return env


# def generate_trajectory(env, model, num_frames=20):
#     """Simulates the trajectory predicted by the model in the environment.

#     Args:
#         model (_type_): Trained model to predict actions at each timestep.
#         env_config (_type_): Config for the environment
#         num_frames (int, optional): Number of steps to predict. Defaults to 20.

#     Returns:
#         _type_: _description_
#     """
#     frames = []
#     actions = []

#     for i in range(num_frames):
#         # Get current frame
#         curr_frame = env.render(mode="rgb_array")

#         obs = env.get_obs()
#         # unsqueeze to emulate batch size of 1
#         imgs = {
#             "rgb_static": preprocess_image(obs["rgb_obs"]["rgb_static"]).unsqueeze(0),
#             "rgb_gripper": preprocess_image(obs["rgb_obs"]["rgb_gripper"]).unsqueeze(0),
#         }
#         robot_obs = torch.tensor(obs["robot_obs"]).float().unsqueeze(0)
#         # Generate action at current frame
#         action = model(imgs, robot_obs)

#         # Save to lists
#         frames.append(curr_frame)
#         actions.append(action)

#         # Force gripper to binary -1 or 1
#         action[0][6] = -1 if action[0][6] < 0 else 1

#         action_dict = {"type": "cartesian_rel", "action": action.detach().squeeze(0)}
#         # Step env
#         observation, reward, done, info = env.step(action_dict)

#         if i % 10 == 0:
#             print("Processed frame", i)

#     return frames, actions


def model2_pred_action(model, obs):
    """Extract relevant information from environment observation and feed to model, return model output.

    Args:
        obs (_type_): _description_
    """
    # unsqueeze to emulate batch size of 1
    imgs = {
        "rgb_static": preprocess_image(obs["rgb_obs"]["rgb_static"]).unsqueeze(0),
        "rgb_gripper": preprocess_image(obs["rgb_obs"]["rgb_gripper"]).unsqueeze(0),
    }
    robot_obs = torch.tensor(obs["robot_obs"]).float().unsqueeze(0)
    return model(imgs, robot_obs)


def main():
    dataset_name = "task_D_D"
    task_name = "stack_block"

    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_name + "/training/"
    val_data_path = HOBBES_DATASET_ROOT_PATH + dataset_name + "/validation/"

    for train_or_val in ("train", "val"):
        if train_or_val == "train":
            data_path = train_data_path
            # continue
        elif train_or_val == "val":
            data_path = val_data_path
        print("using dataset", data_path)

        with open(f"../eval_data/model2_eval-{dataset_name}-{train_or_val}-{task_name}.csv", "w", newline="") as csvfile:
            my_writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

            # set up the start frame for the episode
            episode_ranges = get_task_timeframes(task_name, data_path, 10)

            for episode_range in episode_ranges:
                print(
                    f"Using episode in the range(s): {episode_range} for task {task_name} in {train_or_val} dataset {dataset_name}"
                )

                # load model
                model = Model2.load_from_checkpoint(
                    "../checkpoints/v2/task_D_D/" + task_name + "/trained_model-epoch=999.ckpt"
                )

                # Generate enough trajectory for whichever of our eval metrics needs the most
                num_frames = 3 * (episode_range[1] - episode_range[0])

                demo_obs, demo_actions, predicted_env_states, predicted_actions = all_trajectory_info(
                    action_predictor=lambda obs: model2_pred_action(model, obs), episode_range=episode_range, data_path=data_path, num_frames=num_frames
                )

                action_loss = compute_action_loss(demo_actions, predicted_actions[: len(demo_actions)]).detach().numpy()
                joint_loss = compute_joint_loss(demo_obs, predicted_env_states[: len(demo_obs)]).detach().numpy()
                success = task_success(predicted_env_states, task_name)

                my_writer.writerow(
                    [
                        dataset_name,
                        train_or_val,
                        task_name,
                        episode_range[0],
                        episode_range[1],
                        action_loss,
                        joint_loss,
                        success,
                    ]
                )
                successful = success >= 1
                print(colored(f"Action loss for trajectory of {len(demo_obs)} timesteps: {action_loss}", 'yellow'))
                print(colored(f"Joint loss for trajectory of {len(demo_obs)} timesteps: {joint_loss}", 'yellow'))
                print(colored(f"Within {num_frames} timesteps: " + ("SUCCESSFUL" if successful else "FAILED"), ("green" if successful else "red")))

                # gif generation for targets and predicted
                target_images = [ep["rgb_static"] for ep in demo_obs]
                predicted_images = [obs["rgb_obs"]["rgb_static"] for obs, info in predicted_env_states]
                save_gif(
                    target_images,
                    file_name=f"model2-eval-gifs/model2-{train_or_val}-({task_name})-({episode_range[0]}-{episode_range[1]})-target.gif",
                )
                save_gif(
                    predicted_images[:success + 5],
                    file_name=f"model2-eval-gifs/model2-{train_or_val}-({task_name})-({episode_range[0]}-{episode_range[1]})-prediction.gif",
                )

        # ep = np.load(get_episode_path(episode_range[0], data_path))

        # data = collect_frames(episode_range[0], episode_range[1], data_path)
        # target_actions = [y for x, y in data]

        # # initialize env
        # env = initialize_env(ep["robot_obs"], ep["scene_obs"])

        # # generate sample trajectory
        # desired_num_frames = len(target_actions)
        # frames, actions = generate_trajectory(env, model, desired_num_frames)

        # loss = nn.MSELoss(reduction="mean")

        # target_actions = torch.stack(target_actions[:desired_num_frames])
        # actions = torch.stack(actions).squeeze(1)


if __name__ == "__main__":
    main()
