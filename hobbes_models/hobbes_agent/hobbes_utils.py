from torchvision.transforms import ToPILImage
import numpy as np
from typing import Callable, Any
import torch
import cv2
from typing import List, Tuple, Dict, Callable
import hydra
from old_models.model2_eval import generate_trajectory_full_observations
from torch import nn
from pathlib import Path
from omegaconf import OmegaConf


def get_episode_path(frame_num: int, dataset_path: str, pad_len: int = 7):
    padded_num = str(frame_num).rjust(pad_len, "0")
    return dataset_path + "episode_" + padded_num + ".npz"


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Prepares a 200 x 200 x 3 RGB image for input into the model. Normalizes to [0, 1] and transposes to [3 x 200 x 200]

    Args:
        img (np.ndarray): Raw scene image.

    Returns:
        torch.Tensor: Transposed image ready to be fed into image encoder.
    """
    return torch.from_numpy(np.transpose(img / 255, (2, 0, 1))).float()


def save_gif(frames, file_name="sample.gif"):
    toPilImage = ToPILImage()
    pil_frames = [toPilImage(frame) for frame in frames]
    pil_frames[0].save(
        "/home/grail/willaria_research/hobbes/hobbes_models/hobbes_agent/recordings/" + file_name,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=50,
        loop=0,
    )


def collect_frames(
    start: int,
    stop: int,
    dataset_path: str,
    observation_extractor: Callable = lambda ep: static_image_extractor(ep),
    action_type: str = "rel_actions",
):
    """Loads the specified range of episodes (semi-closed) for creating a demonstration video to imitate.

    Args:
        start (_type_): _description_
        stop (_type_): _description_

    Returns:
        _type_: List of pairs of static camera images and action taken at that step.
    """
    observations = []
    actions = []
    for i in range(start, stop):
        ep = np.load(get_episode_path(i, dataset_path))
        observations.append(observation_extractor(ep))
        actions.append(torch.tensor(ep[action_type]).float())  # 'actions' or 'rel_actions'
    return observations, actions


def static_image_extractor(ep):
    return preprocess_image(ep["rgb_static"])


def get_task_timeframes(target_task_name: str, dataset_path: str, num_demonstrations: int) -> list:
    """Returns the start and end of every episode corresponding to the specified task up to given number of demonstrations.

    Args:
        task_name (str): Name of the task to extract.
        dataset_path (str): Path to the dataset

    Returns:
        list: List of pairs (episode_start_index, episode_end_index)
    """
    # Use built-in language annotations to determine start and stop of demonstration episode
    lang_npy = np.load(dataset_path + "lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    lang = dict(enumerate(lang_npy.flatten(), 1))[1]

    target_task_indxs = []

    task_names = lang["language"]["task"]
    task_frame_ranges = lang["info"]["indx"]

    # Looking for the desired task, collecting frame ranges
    for (task_name, task_range) in zip(task_names, task_frame_ranges):
        if task_name == target_task_name:
            target_task_indxs.append(task_range)
        if len(target_task_indxs) == num_demonstrations:
            break
    if len(target_task_indxs) < num_demonstrations:
        print(f"Warning: Requested {num_demonstrations} demonstrations but only found {len(target_task_indxs)}")
    return target_task_indxs


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


def all_trajectory_info(action_predictor: Callable, ep_range: Tuple, data_path: str, num_frames) -> float:
    """Computes loss for an entire predicted trajectory, comparing with the demonstration, according to loss_func

    Args:
        action_predictor (_type_): Given an episode, return the predicted action. Likely a call to a model.
        ep_range (_type_): First and last episode in the demonstration
        data_path (_type_): Path to folder containing all demonstration episodes
        loss_func (Callable): Compute loss from demonstration states, actions, and predicted states, actions

    Returns:
        _type_: trajectory loss
    """
    data = collect_frames(ep_range[0], ep_range[1], data_path)
    demo_obs = [ob for ob, act in data]
    demo_actions = [act for ob, act in data]
    first_obs = demo_obs[0]
    init_environment = initialize_env(first_obs["robot_obs"], first_obs["scene_obs"])

    predicted_env_states, predicted_actions = generate_trajectory_full_observations(
        init_environment, action_predictor, num_frames
    )

    # predicted_env_state has tuples of (obs, info)
    # predicted_env_observations = [state[0] for state in predicted_env_states]

    return demo_obs, demo_actions, predicted_env_states, predicted_actions


def get_task_success_func(task_str):
    return lambda do, da, pe, pa: task_success(pe, task_str)


def task_success(predicted_env_states, task_str: str):
    conf_dir = Path(__file__).absolute().parents[3] / "calvin_models/calvin_agent/conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    start_info = predicted_env_states[0][0]
    for _, current_info in predicted_env_states:
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task_str})
        if len(current_task_info) > 0:
            print("SUCCESS")
            return 1
    print("FAIL")
    return 0


def compute_joint_loss(demo_obs, predicted_env_states):
    """
    Computes loss by comparing joint positions at each frame from proprioceptive data.
    """
    predicted_env_observations = [state[0] for state in predicted_env_states]

    predicted_joint_positions = torch.stack([obs["robot_obs"] for obs in predicted_env_observations])
    actual_joint_positions = torch.stack([obs["robot_obs"] for obs in demo_obs])
    loss = nn.MSELoss(reduction="mean")
    return loss(input=actual_joint_positions, target=predicted_joint_positions)


def compute_action_loss(demo_actions, predicted_actions):
    """
    Computes loss by comparing action vectors at each frame from proprioceptive data.
    """
    loss = nn.MSELoss(reduction="mean")

    demo_actions_tensor = torch.stack(demo_actions)
    pred_actions_tensor = torch.stack(predicted_actions).squeeze(1)
    return loss(input=pred_actions_tensor, target=demo_actions_tensor)


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


def generate_trajectory_full_observations(env, action_predictor, num_frames=20):
    """Simulates the trajectory predicted by the model in the environment.

    Args:
        env (_type_): Environment initialized to starting frame for task.
        action_predictor (_type_): Predict the next action given an environment observation.
        num_frames (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    environment_states = []
    predicted_actions = []

    for i in range(num_frames):
        env_state = (env.get_obs(), env.get_info())
        environment_states.append(env_state)

        # Generate action at current frame
        action = action_predictor(env_state[0])
        predicted_actions.append(action)

        # Force gripper to binary -1 or 1
        action[0][6] = -1 if action[0][6] < 0 else 1

        action_dict = {"type": "cartesian_rel", "action": action.detach().squeeze(0)}
        # Step env
        env.step(action_dict)

        if i % 10 == 0:
            print("Processed frame", i)

    return environment_states, predicted_actions
