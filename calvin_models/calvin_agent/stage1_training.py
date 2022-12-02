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
from torch.nn import functional as F
import hydra
from hydra import initialize, compose
import cv2

# NOTE: Run from "calvin_models/calvin_agent/", "conda activate calvin_conda_env"

dataset_path = '/home/grail/willaria_research/calvin/dataset/calvin_debug_dataset/training/'

def collect_frames(start, stop):
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
      ep = np.load(dataset_path + "episode_0" + str(i) + ".npz")
      images.append(preprocess_image(ep["rgb_static"]))
      actions.append(torch.from_numpy(np.asarray(ep["actions"])).float()) # 'actions' or 'rel_actions'
  return list(zip(images, actions))

def preprocess_image(img):
  """Prepares a 200 x 200 x 3 RGB image for input into the model. Normalizes to [0, 1] and transposes to [3 x 200 x 200]

  Args:
      img (_type_): _description_

  Returns:
      _type_: _description_
  """
  return torch.from_numpy(np.transpose(img / 255, (2, 0, 1))).float()

def train_model():
  lang = np.load(dataset_path + "lang_annotations/auto_lang_ann.npy", allow_pickle=True)
  lang = dict(enumerate(lang.flatten(), 1))[1]

  model = Stage1Model()
  task_idx = 8
  task_episode_ranges = lang['info']['indx'][task_idx]

  dataset = collect_frames(task_episode_ranges[0], task_episode_ranges[1])
  data_loader = DataLoader(dataset, batch_size=16)

  trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=0.5, default_root_dir='./model_params')
  trainer.fit(model, data_loader)
  return model

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
    env_config.env["use_egl"] = False
    env_config.env["show_gui"] = False
    env_config.env["use_vr"] = False
    env_config.env["use_scene_info"] = True
  return env_config

def main():
  # load model
  model = Stage1Model.load_from_checkpoint("./model_params/lightning_logs/version_0/checkpoints/epoch=999-step=1999.ckpt")
  env_config = create_evaluation_env_config()
  frames = build_frames(model, env_config)
  display_frames(frames)

if __name__ == "__main__":
  main()