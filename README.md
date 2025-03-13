# DT2112
This repo contains the demo and some of the soruce code for the "Identifying speakers using voice embeddings", DT2112 Project.

This was built by:
- Filip Danielsson, fidani@kth.se 
- Jan Cáp, jcap@kth.se 
- Viktor Ronnbacka Nyback, viktorrn@kth.se


## Installation
To install the package, run:
```bash
make install_dev
```
It will install the requirements and the package itself in development mode.
Also it will install pre-commit hooks to ensure code quality.

## Downloading the data

To download the data, run notebook `notebooks/load_data.ipynb`. It will download the data and save it to `data/` directory.

## Running the demo
To be able to run the demo, you need following files:
- `data/vox2_meta.csv` - metadata file for the VoxCeleb2 dataset
- `data/celebs_dev.csv` - metadata file for the Celeb subset of VoxCeleb2 dataset containing only 200 most famous people and links to their longest clips
- `data/filtered_celebs_data/celebs_200_9_clips/` - directory with the vector database


run the following command to run the demo:
```bash
python src/demo/app.py
```
