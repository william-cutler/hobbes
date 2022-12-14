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
from single_task_imitator import SingleTaskImitator
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


def model2_pred_action(model, obs,
                       rgb_static=False,
                       depth_static=False,
                       rgb_gripper=False,
                       depth_gripper=False,
                       proprioceptive=False,
                       rgb_tactile=False,
                       depth_tactile=False,):
    """Extract relevant information from environment observation and feed to model, return model output.

    Args:
        obs (_type_): _description_
    """
    static_img = []
    if rgb_static:
        static_img.append(preprocess_image(obs["rgb_obs"]['rgb_static']))
    if depth_static:
        static_img.append(preprocess_image(np.expand_dims(obs["depth_obs"]['depth_static'], axis=2), scale=10))
    static_img = torch.cat(static_img, dim=0).unsqueeze(0) if len(static_img) > 0 else []

    gripper_img = []
    if rgb_gripper:
        gripper_img.append(preprocess_image(obs["rgb_obs"]['rgb_gripper']))
    if depth_gripper:
        gripper_img.append(preprocess_image(np.expand_dims(obs["depth_obs"]['depth_gripper'], axis=2), scale=1))
    gripper_img = torch.cat(gripper_img, dim=0).unsqueeze(0) if len(gripper_img) > 0 else []
    
    robot_obs = []
    if proprioceptive:
        robot_obs = torch.tensor(obs['robot_obs']).float().unsqueeze(0)

    imgs = {
        'static': static_img,
        'gripper': gripper_img,
        'proprioceptive': robot_obs
    }
    
    return model(imgs)

def generate_predicted_target_gifs():
    dataset_name = "task_D_D"
    task_name = "stack_block"
    sensor_combination_string = "rgb_static"
    
    train_or_val = "val"
    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_name + "/training/"
    val_data_path = HOBBES_DATASET_ROOT_PATH + dataset_name + "/validation/"
    data_path = train_data_path if train_or_val == "train" else val_data_path
    
    episode_ranges = [[36177,36220], [13725,13789]]
    
    rgb_static = True
    depth_static = True
    rgb_gripper = True
    depth_gripper = True
    proprioceptive = True
    rgb_tactile = False
    depth_tactile = False
    
    # load model
    model = SingleTaskImitator.load_from_checkpoint(
        "../checkpoints/v2/" + dataset_name + "/" + task_name + "/sensors/" + sensor_combination_string + "/latest-epoch=99.ckpt",
        rgb_static=rgb_static,
        depth_static=depth_static,
        rgb_gripper=rgb_gripper,
        depth_gripper=depth_gripper,
        proprioceptive=proprioceptive,
        rgb_tactile=rgb_tactile,
        depth_tactile=depth_tactile,
    )
    
    for episode_range in episode_ranges:
        # Generate enough trajectory for whichever of our eval metrics needs the most
        num_frames = 3 * (episode_range[1] - episode_range[0])

        demo_obs, demo_actions, predicted_env_states, predicted_actions = all_trajectory_info(
            action_predictor=lambda obs: model2_pred_action(
                model, obs,
                rgb_static=rgb_static,
                depth_static=depth_static,
                rgb_gripper=rgb_gripper,
                depth_gripper=depth_gripper,
                proprioceptive=proprioceptive,
                rgb_tactile=rgb_tactile,
                depth_tactile=depth_tactile,
            ),
            episode_range=episode_range,
            data_path=data_path,
            num_frames=num_frames
        )
        
        target_images = [ep["rgb_static"] for ep in demo_obs]
        predicted_images = [obs["rgb_obs"]["rgb_static"] for obs, info in predicted_env_states]
        save_gif(
            target_images,
            file_name=f"sensor-eval-gifs/" + sensor_combination_string + "/model2-{train_or_val}-({task_name})-({episode_range[0]}-{episode_range[1]})-target.gif",
        )
        save_gif(
            predicted_images,
            file_name=f"sensor-eval-gifs/" + sensor_combination_string + "/model2-{train_or_val}-({task_name})-({episode_range[0]}-{episode_range[1]})-prediction.gif",
        )

def main():
    ################ parameters ################
    dataset_name = "task_D_D"
    task_name = "stack_block"
    sensor_combination_string = "rgb_static"

    rgb_static = True
    depth_static = False
    rgb_gripper = False
    depth_gripper = False
    proprioceptive = False
    rgb_tactile = False
    depth_tactile = False
    
    gif_generation = False
    ################ parameters ################
    

    train_data_path = HOBBES_DATASET_ROOT_PATH + dataset_name + "/training/"
    val_data_path = HOBBES_DATASET_ROOT_PATH + dataset_name + "/validation/"

    for train_or_val in ("train", "val"):
        if train_or_val == "train":
            data_path = train_data_path
        elif train_or_val == "val":
            data_path = val_data_path
            continue
        print("using dataset", data_path)

        with open(f"../eval_data/sensor_combination_tests/model2_eval-{dataset_name}-{train_or_val}-{task_name}-" + sensor_combination_string + "-more-epochs.csv", "w", newline="") as csvfile:
            my_writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

            # set up the start frame for the episode
            episode_ranges = get_task_timeframes(task_name, data_path, 1000)

            for episode_range in episode_ranges:
                print(
                    f"Using episode in the range(s): {episode_range} for task {task_name} in {train_or_val} dataset {dataset_name}"
                )

                # load model
                model = SingleTaskImitator.load_from_checkpoint(
                    "../checkpoints/v2/" + dataset_name + "/" + task_name + "/sensors/" + sensor_combination_string + "/latest-epoch=199.ckpt",
                    rgb_static=rgb_static,
                    depth_static=depth_static,
                    rgb_gripper=rgb_gripper,
                    depth_gripper=depth_gripper,
                    proprioceptive=proprioceptive,
                    rgb_tactile=rgb_tactile,
                    depth_tactile=depth_tactile,
                )

                # Generate enough trajectory for whichever of our eval metrics needs the most
                num_frames = 3 * (episode_range[1] - episode_range[0])

                demo_obs, demo_actions, predicted_env_states, predicted_actions = all_trajectory_info(
                    action_predictor=lambda obs: model2_pred_action(
                        model, obs,
                        rgb_static=rgb_static,
                        depth_static=depth_static,
                        rgb_gripper=rgb_gripper,
                        depth_gripper=depth_gripper,
                        proprioceptive=proprioceptive,
                        rgb_tactile=rgb_tactile,
                        depth_tactile=depth_tactile,
                    ),
                    episode_range=episode_range,
                    data_path=data_path,
                    num_frames=num_frames
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
                if gif_generation:
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



if __name__ == "__main__":
    main()
    #generate_predicted_target_gifs()
