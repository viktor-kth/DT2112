import tempfile
from pathlib import Path

import librosa
import pandas as pd
from flask import Flask, jsonify, render_template, request

from kth_sr import FAISS, embeddings, first_k_windows, loaddata

# Note to run this file do

app = Flask(__name__)

SAMPLE_RATE = 16000
NUM_FRAMES = 160

model = embeddings.get_embedding_model()
storage = FAISS.load("./data/filtered_celebs_data/celebs_200_9_clips")
celeb_information = loaddata.get_celeb_data(
    "./data/filtered_celebs_data/celebs_200_9_clips"
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/findmatch", methods=["POST"])
def findmatch():
    file = request.files["audio"]
    print("Received file:", file.filename, file.content_type)  # Debug info

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = Path(temp_dir) / "temp_audio.wav"

        # Save and process the audio file
        file.save(temp_wav_path)

        # Load audio data with librosa and resample to required sample rate
        try:
            audio_data, _ = librosa.load(str(temp_wav_path), sr=SAMPLE_RATE, mono=True)
            mfcc_features = first_k_windows(audio_data, SAMPLE_RATE, NUM_FRAMES, k=100)
            embedd = model.m.predict(mfcc_features, verbose=None)
            distances, metadata = storage.search(embeddings=embedd, k=6)
            filterd_dev = celeb_information[
                celeb_information["VoxCeleb2_ID"].isin(metadata[0])
            ]

            order_df = pd.DataFrame(
                {"VoxCeleb2_ID": metadata[0], "order": range(len(metadata[0]))}
            )

            filterd_dev = (
                pd.merge(filterd_dev, order_df, on="VoxCeleb2_ID")
                .sort_values("order")
                .drop("order", axis=1)
            )

            return jsonify(
                {
                    "status": 200,
                    "message": "File processed successfully",
                    "distances": distances.tolist(),
                    "data": filterd_dev.to_dict(orient="records"),
                }
            )

        except Exception as e:
            print("Error processing audio:", str(e))  # Debug info
            return jsonify({"status": 500, "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
