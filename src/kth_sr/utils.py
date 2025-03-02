from pathlib import Path
import pandas as pd


def get_df_by_downloaded_folder(folder: str) -> pd.DataFrame:
    """Create a DataFrame from folder with downloaded mp3 files.

    Args:
        folder (str): path to folder with downloaded mp3 files.
    """
    # iterate all files in the folder and get their length
    data = []

    for file in Path(folder).rglob("*.mp3"):
        user_id = file.stem.split("_")[0]  # user_id in voxceleb2
        duration_ms = int(file.stem.split("_")[2])  # duration in ms

        # Convert to seconds
        duration_s = duration_ms / 1000.0
        data.append({"speaker": user_id, "duration_s": duration_s})

    return pd.DataFrame(data)
