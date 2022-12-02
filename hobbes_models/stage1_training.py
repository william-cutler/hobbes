from calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv
import numpy as np
import torch
from torch.utils.data import DataLoader
from stage1_model import Stage1Model
import pytorch_lightning as pl
from torch.nn import functional as F
import hydra
from hydra import initialize, compose
import cv2

dataset_path = '/home/grail/willaria_research/calvin/dataset/calvin_debug_dataset/training/'

lang = np.load(dataset_path + "lang_annotations/auto_lang_ann.npy", allow_pickle=True)
lang = dict(enumerate(lang.flatten(), 1))[1]

def collect_frames(start, stop):
    images = []
    actions = []
    for i in range(start, stop):
        ep = np.load(dataset_path + "episode_0" + str(i) + ".npz")
        images.append(torch.from_numpy(np.transpose(ep["rgb_static"] / 255, (2, 0, 1))).float())
        actions.append(torch.from_numpy(np.asarray(ep["actions"])).float()) # 'actions' or 'rel_actions'
    return list(zip(images, actions))

# data
task_idx = 8
dataset = collect_frames(lang['info']['indx'][task_idx][0], lang['info']['indx'][task_idx][1])
data_loader = DataLoader(dataset, batch_size=16)

# if we want train test split
# mnist_train, mnist_val = random_split(dataset, [55000, 5000])
# train_loader = DataLoader(mnist_train, batch_size=32)
# val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = Stage1Model()

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=0.5)
trainer.fit(model, data_loader)

trainer.test(model, data_loader)

with initialize(config_path="./calvin/calvin_env/conf/"):
  cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
  cfg.env["use_egl"] = False
  cfg.env["show_gui"] = False
  cfg.env["use_vr"] = False
  cfg.env["use_scene_info"] = True
  print(cfg.env)

def build_frames(model, start_frame):
    env = hydra.utils.instantiate(cfg.env)
    action = model(start_frame)
    # TODO: Figure out how to instantiate environment at particular timestep

# observation = env.reset()
# #The observation is given as a dictionary with different values
# print(observation.keys())
for i in range(5):
  # The action consists in a pose displacement (position and orientation)
  action_displacement = np.random.uniform(low=-10, high=10, size=6)
  # And a binary gripper action, -1 for closing and 1 for oppening
  action_gripper = np.random.choice([-1, 1], size=1)
  action = np.concatenate((action_displacement, action_gripper), axis=-1)
  observation, reward, done, info = env.step(action)
  rgb = env.render(mode="rgb_array")[:,:,::-1]
  cv2.imshow("Environment View", rgb)
  cv2.waitKey(0)
cv2.destroyAllWindows()