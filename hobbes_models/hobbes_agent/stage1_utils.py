from torchvision.transforms import ToPILImage
import numpy as np
import torch

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
    pil_frames[0].save('recordings/' + file_name,
               save_all=True, append_images=pil_frames[1:], optimize=False, duration=50, loop=0)