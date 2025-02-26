import numpy as np
from deep_speaker.audio import pad_mfcc, mfcc_fbank
import librosa


def first_k_windows(audio, sample_rate: int, num_frames: int, k: int = 5):
    """Return first k windows of the Mel-filterbank energy features.
    Args:
        audio (np.ndarray): Audio array.
        sample_rate (int): Sample rate of the audio.
        num_frames (int): Number of frames in each window.
        k (int): Number of frames to return.

    Returns:
        np.ndarray: Array of first k frames prepared for the model.
        Dimensions are (k, num_frames, 64, 1)
    """

    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    # left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
    # right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate
    # TODO: could use trim_silence() here or a better VAD.
    audio_voice_only = audio[offsets[0] : offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)

    segments = []
    for i in range(k):
        start_frame = i * num_frames
        end_frame = start_frame + num_frames

        if end_frame > mfcc.shape[0]:
            padded = pad_mfcc(mfcc[i * num_frames :], num_frames)
            segments.append(padded)
            break

        segments.append(mfcc[start_frame:end_frame])

    return np.array(segments).reshape((-1, 160, 64, 1))


def first_k_windows_from_file(
    file_path: str, sample_rate: int, num_frames: int, k: int = 5
):
    """Return first k windows of the Mel-filterbank energy features from a file.
    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Sample rate of the audio.
        num_frames (int): Number of frames in each window.
        k (int): Number of frames to return.

    Returns:
        np.ndarray: Array of first k frames prepared for the model.
        Dimensions are (k, num_frames, 64, 1)
    """
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return first_k_windows(audio, sample_rate, num_frames, k)
