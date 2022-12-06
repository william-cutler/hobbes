from torchvision.transforms import ToPILImage
import numpy as np
from typing import Callable, Any
import torch
import cv2
from typing import List, Tuple, Dict, Callable
import hydra
from old_models.model2_eval import generate_trajectory_full_observations
from torch import nn


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


def save_gif(frames, file_name='sample.gif'):
    toPilImage = ToPILImage()
    pil_frames = [toPilImage(frame) for frame in frames]
    pil_frames[0].save('/home/grail/willaria_research/hobbes/hobbes_models/hobbes_agent/recordings/' + file_name,
               save_all=True, append_images=pil_frames[1:], optimize=False, duration=50, loop=0)

def collect_frames(start: int, stop: int, dataset_path: str, observation_extractor: Callable = lambda ep: static_image_extractor(ep), action_type: str = "rel_actions"):
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
        if (len(target_task_indxs) == num_demonstrations):
            break
    if (len(target_task_indxs) < num_demonstrations):
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


def model_evaluation_trajectory_loss(model, demonstration: List, loss_func: Callable) -> float:
    """_summary_

    Args:
        model (Model2): _description_
        demonstration (List): List of episodes in order in the demonstration.

    Returns:
        float: _description_
    """
    
    
    first_episode = demonstration[0]
    init_environment = initialize_env(first_episode["robot_obs"], first_episode["scene_obs"])
    
    environment_states, predicted_actions = generate_trajectory_full_observations(init_environment, model, len(demonstration))
    return loss_func(demonstration_states, demonstration_actions, environment_states, predicted_actions)


def actions_trajectory_loss(demonstration_states, demonstration_actions, environment_states, predicted_actions):
    loss = nn.MSELoss(reduction='mean')
    
    demo_actions_tensor = torch.stack(demonstration_actions)
    pred_actions_tensor = torch.stack(predicted_actions).squeeze(1)
    return loss(input=pred_actions_tensor, target=demo_actions_tensor)
