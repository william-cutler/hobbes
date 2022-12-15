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
            
            # set constant length for demonstrations
            if len(demonstration) == 64:
                observations.append(demonstration)

                target_actions.append(torch.stack(actions))

        self.observations = observations
        self.actions = target_actions
        self.len = len(observations)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        observations = self.observations[idx]
        # print('length of sequence at idx', idx, 'is', len(observations), 'number of items is', self.len)
        
        rgb_static = []
        rgb_gripper = []
        robot_obs = []
        for observation in observations:
            rgb_static.append(observation.get('rgb_static'))
            rgb_gripper.append(observation.get('rgb_gripper'))
            robot_obs.append(observation.get('robot_obs'))

        actions = self.actions[idx]
        
        # print(demo_idx, idx)
        
        demo = {
            'rgb_static': torch.stack(rgb_static),
            'rgb_gripper': torch.stack(rgb_gripper),
            'robot_obs': torch.stack(robot_obs)
        }
        
        item = {'state_observations': demo,
                'state_observations_num': len(rgb_static),
        }

        """ demo is a dictionary: 
        {
            'rgb_static': list(images),
            'rgb_gripper': list(images),
            'robot_obs': list(robot_obs)
        }
        """
        return item, actions