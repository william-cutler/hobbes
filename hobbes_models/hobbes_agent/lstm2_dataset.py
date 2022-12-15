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

        self.len = len(train_timeframes)

        observations = []
        target_actions = []
        for timeframe in train_timeframes:
            demonstration, actions = collect_frames(
                timeframe[0], timeframe[1], dataset_path, observation_extractor=multi_extractor, action_type="rel_actions")
            observations.append(demonstration)
            target_actions.append(torch.stack(actions))

        self.observations = observations
        self.actions = target_actions

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        demo = self.observations[idx]
        actions = self.observations[idx]
        
        # print(demo_idx, idx)
        
        item = {'state_observations': demo,
                'state_observations_num': len(demo),
        }

        """ demo is a dictionary: 
        {
            'rgb_static': list(images),
            'rgb_gripper': list(images),
            'robot_obs': list(robot_obs)
        }
        """
        return item, actions