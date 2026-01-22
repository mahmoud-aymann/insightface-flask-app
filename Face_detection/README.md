# Face Detection Models

This folder contains the trained models and embeddings for face recognition.

## Contents

- `weights/best.pt` - YOLOv5 face detection model weights
- `embeddings_raw.pkl` - Face embeddings database for recognition

## Model Information

### YOLOv5 Model (`best.pt`)
- **Purpose:** Face detection
- **Input:** Images
- **Output:** Bounding boxes for detected faces
- **Format:** PyTorch model (.pt)

### Embeddings Database (`embeddings_raw.pkl`)
- **Purpose:** Face recognition
- **Format:** Pickle file containing face embeddings
- **Structure:** Dictionary with person names as keys and embeddings as values

## Usage

These models are automatically loaded by `app.py` when the Flask application starts.

## Training Your Own Models

To train your own models:

1. Run `Face_Recognition_Pipeline.ipynb`
2. Follow all 14 steps in the notebook
3. The notebook will generate:
   - `best.pt` in `weights/` folder
   - `embeddings_raw.pkl` in this folder

## Notes

- Sample models are included for testing
- For production, train models on your own dataset
- Model files can be large (tens to hundreds of MB)
