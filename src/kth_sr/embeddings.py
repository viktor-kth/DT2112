from pathlib import Path

import gdown
from deep_speaker.conv_models import DeepSpeakerModel


def get_embedding_model(
    model_out_path: str = "data/ResCNN_triplet_training_checkpoint_265.h5",
) -> DeepSpeakerModel:
    """Load pre-trained embedding model Deep Speaker from Google Drive.

    Args:
        model_out_path (str): Path for the pre-trained model to be saved.

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
