from pathlib import Path
import gdown
from deep_speaker.conv_models import DeepSpeakerModel
from kth_sr.mfcc import first_k_windows
from kth_sr.embeddings.strategies import BaseEmbeddingStrategy, FirstWindowStrategy
import numpy as np


def get_embedding_model(
    model_out_path: str = "data/ResCNN_triplet_training_checkpoint_265.h5",
) -> DeepSpeakerModel:
    """Load pre-trained embedding model Deep Speaker from Google Drive.

    Args:
        model_out_path (str): Path for the pre-trained model to be saved or loaded from.

    Returns:
        DeepSpeakerModel: Pre-trained model.
    """
    if not Path(model_out_path).exists():
        file_url = "https://drive.google.com/uc?id=1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP"
        gdown.download(file_url, model_out_path, quiet=False)

    # Define the model here.
    model = DeepSpeakerModel()

    model.m.load_weights(model_out_path, by_name=True)
    return model


class EmbeddingModel:
    """Class for embedding audio files using a pre-trained model.
    Also allows for combining embeddings from multiple windows using a strategy.

    Args:
        model (DeepSpeakerModel): Pre-trained model for embeddings.
        strategy (BaseEmbeddingStrategy, optional): Used strategy for combining embeddings from multiple windows. Defaults to FirstWindowStrategy().
        num_frames (int, optional): Number of frames in each window. Defaults to 160.
        k (int, optional): Number of windows to consider. Defaults to None. If None, consider all windows.
    """

    model: DeepSpeakerModel
    """Pre-trained model for embeddings."""
    strategy: BaseEmbeddingStrategy
    """Used strategy for combining embeddings from multiple windows."""
    num_frames: int
    """Number of frames in each window."""
    k: int
    """Number of windows to consider."""

    def __init__(
        self,
        model: DeepSpeakerModel,
        strategy: BaseEmbeddingStrategy = FirstWindowStrategy(),
        num_frames: int = 160,
        k: int | None = None,
    ):
        self.model = model
        self.strategy = strategy
        self.num_frames = num_frames

        # If k is None, consider all windows.
        if k is None:
            k = 2**31 - 1
        self.k = k

    def embed(self, audios: list[np.ndarray], sample_rate: int) -> np.ndarray:
        """Embed an audio signals using the pre-trained model to get the embeddings
        and then apply the strategy to combine the embeddings.

        Args:
            audio (list[np.ndarray]): List of audio arrays. Each audio array is a numpy array.
            sample_rate (int): Sample rate of the audio signals.

        Returns:
            np.ndarray: Embedding of the audio.
        """
        windows = self._get_mfcc_windows(audios, sample_rate)
        embeddings = self.model.m.predict(windows)
        return self.strategy.apply(embeddings)

    def embed_batch(
        self, audios: list[list[np.ndarray]], sample_rate: int
    ) -> list[np.ndarray]:
        """Embed a batch of audio files from different speakers.
        Each speaker can have multiple audio files.

        Args:
            audios (list[list[np.ndarray]]): List of speakers. Each speaker is a list of audio arrays.
            sample_rate (int): Sample rate of the audio signals.

        Returns:
            np.ndarray: Embeddings of the audio batch.
        """
        # list of speakers, each speaker is a ndarray of windows
        speaker_windows = []
        for speaker in audios:
            one_speaker_windows = self._get_mfcc_windows(speaker, sample_rate)
            speaker_windows.append(one_speaker_windows)

        # we want to call model.m.predict only once for all windows
        # so we flatten the list of windows
        # but we need to keep track of the speaker boundaries
        speaker_boundaries = np.cumsum([len(speaker) for speaker in speaker_windows])
        windows = np.vstack(speaker_windows)
        embeddings = self.model.m.predict(windows)
        # we split the embeddings back into the original speakers
        speakers_embeds = np.split(embeddings, speaker_boundaries[:-1])
        return [
            self.strategy.apply(one_speaker_embeds)
            for one_speaker_embeds in speakers_embeds
        ]

    def _get_mfcc_windows(
        self, audios: list[np.ndarray], sample_rate: int
    ) -> np.ndarray:
        """Get the Mel-filterbank energy features for each audio.

        Args:
            audios (list[np.ndarray]): List of audio arrays. Each audio array is a numpy array.
            sample_rate (int): Sample rate of the audio signals.

        Returns:
            np.ndarray: Mel-filterbank energy features for each audio.
        """
        all_windows = [
            first_k_windows(audio, sample_rate, self.num_frames, self.k)
            for audio in audios
        ]
        return np.vstack(all_windows)
