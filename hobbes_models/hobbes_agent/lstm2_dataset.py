from torch.utils.data import Dataset
from hobbes_utils import *

def multi_extractor(ep):
    observations = {
        'rgb_static': preprocess_image(ep["rgb_static"]),
        'rgb_gripper': preprocess_image(ep["rgb_gripper"]),
        'robot_obs': torch.tensor(ep['robot_obs']).float()
    }
    return observations

class LSTMDataset(Dataset):
    """Dataset for LSTM model."""

    def __init__(self, task_name: str, dataset_path: str, num_observations: int):
        """_summary_

        Args:
            demonstrations (list of list of frames): list of demonstration videos
        """

        train_timeframes = get_task_timeframes(
            target_task_name=task_name, dataset_path=dataset_path, num_demonstrations=num_observations)

        observations = []
        target_actions = []
        for timeframe in train_timeframes:
            demonstration, actions = collect_frames(
                timeframe[0], timeframe[1], dataset_path, observation_extractor=multi_extractor, action_type="rel_actions")
            observations.append(demonstration)
            target_actions.append(torch.stack(actions))

        self.observations = observations
        self.actions = target_actions
        self.frame_to_demo_num_and_frame_num = {}
        i = 0
        for d in range(len(observations)):
            for f in range(len(observations[d])):
                self.frame_to_demo_num_and_frame_num[i] = (d, f)
                i += 1
        self.len = len(self.frame_to_demo_num_and_frame_num)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        demo_idx, frame_num = self.frame_to_demo_num_and_frame_num[idx]
        demo = self.observations[demo_idx]
        
        # print(demo_idx, idx)
        
        item = {'demonstration_images': demo,
                'demonstration_images_num': len(demo),
                'runtime_image': demo[frame_num],
        }

        return item, self.actions[demo_idx][frame_num]