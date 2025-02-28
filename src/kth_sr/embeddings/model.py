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

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed an audio file using the pre-trained model and strategy.

        Args:
            audio (np.ndarray): Audio signal.
            sample_rate (int): Sample rate of the audio.

        Returns:
            np.ndarray: Embedding of the audio.
        """
        windows = first_k_windows(audio, sample_rate, self.num_frames, self.k)
        embeddings = self.model.m.predict(windows)
        return self.strategy.apply(embeddings)
