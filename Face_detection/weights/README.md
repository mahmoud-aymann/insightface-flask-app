# Model Weights

## Place Your Model Here

Place your trained YOLOv5 face detection model (`best.pt`) in this folder.

## File Structure

```
Face_detection/
└── weights/
    └── best.pt  # Your YOLOv5 model file
```

## How to Get the Model

1. **Train your own model:**
   - Run `Face_Recognition_Pipeline.ipynb`
   - Follow the training steps
   - Export the best weights as `best.pt`

2. **Use pre-trained model:**
   - Download a pre-trained YOLOv5 face detection model
   - Rename it to `best.pt`
   - Place it in this folder

## Notes

- Model file can be large (tens to hundreds of MB)
- The application will automatically load this model on startup
- If the file doesn't exist, you'll see an error message with instructions
