import numpy as np
from deep_speaker.audio import pad_mfcc


def first_k_windows(mfcc: np.ndarray, num_frames: int, k: int = 5):
    """Return first k frames of the mfcc array.

    Args:
        mfcc (np.ndarray): MFCC array of processed audio.
        num_frames (int): Number of frames in each window.
        k (int): Number of frames to return.

    Returns:
        np.ndarray: Array of first k frames prepared for the model.
        Dimensions are (k, num_frames, 64, 1)
    """
    frames = []
    for i in range(k):
        end_frame = (i + 1) * num_frames
        if end_frame < mfcc.shape[0]:
            frames.append(mfcc[i * num_frames : end_frame])
        else:
            padded = pad_mfcc(mfcc[i * num_frames :], num_frames)
            frames.append(padded)
            break
    return np.array(frames).reshape(-1, num_frames, 64, 1)
