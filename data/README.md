# Sample Data

This folder contains sample images for testing the face recognition system.

## Structure

```
data/
├── sample/                    # Original sample images
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── cropped_faces/             # Cropped face images (processed)
    ├── face1.jpg
    ├── face2.jpg
    └── ...
```

## Usage

1. **For Training:**
   - Add your training images organized by person name
   - Run the `Face_Recognition_Pipeline.ipynb` notebook
   - The notebook will process all images and generate embeddings

2. **For Testing:**
   - Use the web interface (`app.py`) to upload test images
   - The system will recognize faces based on trained embeddings

## Notes

- Sample images are included for demonstration
- For production use, add your own dataset
- Ensure images are clear and contain visible faces
- Recommended: 10-50 images per person for good accuracy
