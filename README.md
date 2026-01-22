# Face Recognition System

> **Note:** This project is part of a graduation project.

## 🎥 Demo Video

Watch the demo video to see the face recognition system in action:

[![Face Recognition Demo](https://img.youtube.com/vi/N2aGgLsVP3Y/0.jpg)](https://youtu.be/N2aGgLsVP3Y)

*Click the image above to watch the demo video on YouTube*

## Overview

A complete face recognition system using Flask, YOLOv5, and InsightFace. This project provides a simple and professional web interface for testing face recognition, along with a complete training pipeline in Jupyter Notebook.

The system can detect faces in images and recognize them based on a pre-trained database of face embeddings. It features a modern web interface with real-time recognition capabilities and adjustable confidence thresholds.

## Features

- ✅ Modern and user-friendly web interface
- ✅ Real-time face recognition
- ✅ Adjustable recognition threshold (0.3 - 0.9)
- ✅ System statistics dashboard
- ✅ Drag and drop image upload
- ✅ Responsive design (mobile-friendly)
- ✅ Complete training pipeline (Jupyter Notebook)
- ✅ Sample data included for testing
- ✅ Pre-trained models available on Google Drive

## Project Structure

```
insightface_github_complete/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── Face_Recognition_Pipeline.ipynb # Complete training pipeline (14 steps)
├── demo_video.mp4                  # Demo video
├── SETUP.md                        # Setup instructions for model files
├── static/
│   ├── script.js                  # Frontend JavaScript
│   └── style.css                  # Frontend styles
├── templates/
│   └── index.html                 # Main HTML template
├── Face_detection/                 # Models and embeddings (generated)
│   ├── weights/
│   │   └── best.pt                # YOLOv5 weights (download from Google Drive)
│   ├── embeddings_raw.pkl         # Face embeddings database (download from Google Drive)
│   └── insightface/               # InsightFace models (auto-downloaded)
│       └── buffalo_l/             # ONNX models
├── data/                          # Sample training data
│   ├── sample/                    # Original sample images (5 persons)
│   └── cropped_faces/             # Cropped face images
└── documentation/                 # Additional documentation
```

## Requirements

### Python Packages
```bash
pip install -r requirements.txt
```

### System Requirements
- Python 3.8+
- Visual C++ Redistributable 2019 (for Windows)
  - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Jupyter Notebook (for training pipeline)
  ```bash
  pip install jupyter notebook
  ```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mahmoud-aymann/insightface-flask-app.git
   cd insightface-flask-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get model files** (Required before running the app)
   
   **Option A: Download pre-trained models (Recommended)**
   - Download from [Google Drive](https://drive.google.com/drive/folders/1uUGziBy7ArD8kcmJk_56Y_W0NUsn-76U?usp=sharing)
   - Extract and place files:
     - `best.pt` → `Face_detection/weights/best.pt`
     - `embeddings_raw.pkl` → `Face_detection/embeddings_raw.pkl`
     - `insightface/` → `Face_detection/insightface/`
   - See [SETUP.md](SETUP.md) for detailed instructions
   
   **Option B: Generate from code**
   - Open `Face_Recognition_Pipeline.ipynb` in Jupyter Notebook
   - Run all cells sequentially (14 steps)
   - The notebook will generate all required model files
   - InsightFace models will download automatically on first use

## Usage

### Web Application

1. **Run the Flask application**
   ```bash
   python app.py
   ```

2. **Open browser**
   - Navigate to: http://127.0.0.1:5000

3. **Use the interface**
   - Upload an image containing a face (drag & drop or click to browse)
   - Adjust threshold slider (default: 0.45)
     - Lower values = more permissive (may have false positives)
     - Higher values = more strict (may miss some matches)
   - Click "Recognize Face"
   - View recognition results:
     - Person name (if recognized)
     - Confidence score (0-100%)
     - Status (Recognized/Unknown)

### Training Pipeline

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Run the training pipeline**
   - Open `Face_Recognition_Pipeline.ipynb`
   - Follow the 14-step guide:
     - Steps 1-2: Install packages
     - Steps 3-4: Configure paths and load YOLOv5
     - Steps 5-7: Detect and crop faces
     - Steps 8-10: Extract embeddings with InsightFace
     - Steps 11-14: Build database and test

## Sample Data

This repository includes sample data for testing:
- **Sample images** in `data/sample/` folder (5 persons, 3 images each)
- **Cropped faces** in `data/cropped_faces/` folder (processed images)

You can use this data to test the system or add your own images organized by person name.

## Model Files

Model files are **not included** in this repository because they are large (>100MB total). You have two options:

1. **Download pre-trained models** from [Google Drive](https://drive.google.com/drive/folders/1uUGziBy7ArD8kcmJk_56Y_W0NUsn-76U?usp=sharing) (Recommended - fastest way)
2. **Generate from code** by running `Face_Recognition_Pipeline.ipynb` (Takes time but gives you full control)

See [SETUP.md](SETUP.md) for detailed instructions on both methods.

## Technologies Used

- **Backend:** Flask, Python
- **Face Detection:** YOLOv5 (PyTorch)
- **Face Recognition:** InsightFace
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **ML Libraries:** scikit-learn, NumPy, OpenCV
- **Notebook:** Jupyter Notebook

## Important Notes

1. **Model Files:** Model files (`best.pt` and `embeddings_raw.pkl`) are **not included** in this repository. Download them from Google Drive or generate using the training pipeline. See [SETUP.md](SETUP.md).

2. **Environment:** Make sure to install Visual C++ Redistributable 2019 before running the application on Windows. This is required for onnxruntime.

3. **Security:** This application is intended for local/development use. Do not use it in production without proper security measures (authentication, rate limiting, etc.).

4. **Training:** Use the provided Jupyter Notebook (`Face_Recognition_Pipeline.ipynb`) to train models on your own dataset. The notebook contains 14 steps with clear instructions.

5. **Threshold:** The default threshold is 0.45. Adjust it based on your needs:
   - **0.3-0.4:** Very permissive (may recognize wrong people)
   - **0.45-0.5:** Balanced (recommended)
   - **0.6-0.7:** Strict (may miss some matches)
   - **0.8+:** Very strict (only high confidence matches)

## Troubleshooting

### Issue: "onnxruntime DLL error"
**Solution:**
1. Install Visual C++ Redistributable 2019 from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Restart your terminal/IDE completely
3. Reinstall onnxruntime:
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime
   ```
4. If still not working, try CPU-only version:
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-cpu
   ```

### Issue: "No module named 'yolov5'"
**Solution:**
```bash
pip install yolov5
```

### Issue: "No module named 'insightface'"
**Solution:**
```bash
pip install insightface
```

### Issue: "No face detected"
**Solution:** 
- Make sure the image contains a clear and visible face
- Face should be facing forward (not side profile)
- Good lighting helps detection accuracy
- Try different images

### Issue: Models not loading
**Solution:** 
- Check that `best.pt` exists in `Face_detection/weights/`
- Check that `embeddings_raw.pkl` exists in `Face_detection/`
- Download models from [Google Drive](https://drive.google.com/drive/folders/1uUGziBy7ArD8kcmJk_56Y_W0NUsn-76U?usp=sharing) or run the training pipeline
- See [SETUP.md](SETUP.md) for detailed instructions

### Issue: "Unknown" recognition for known faces
**Solution:**
- Lower the threshold (try 0.4 or 0.35)
- Add more training images for that person
- Ensure training images have good quality and variety
- Re-run the training pipeline to update embeddings

## Training Your Own Models

1. **Prepare your dataset**
   - Organize images in folders by person name
   - Example: `data/person1/`, `data/person2/`, etc.
   - Recommended: 10-50 images per person
   - Ensure images are clear and contain visible faces

2. **Run the training pipeline**
   - Open `Face_Recognition_Pipeline.ipynb`
   - Update paths in Step 3 if needed
   - Run all cells sequentially (Steps 1-14)
   - The notebook will generate:
     - Trained YOLOv5 model (`best.pt`)
     - Face embeddings database (`embeddings_raw.pkl`)

3. **Test your models**
   - Run `python app.py` to start the web interface
   - Upload test images to verify recognition accuracy
   - Adjust threshold as needed

## Performance Tips

- **Faster recognition:** Use GPU if available (modify InsightFace providers)
- **Better accuracy:** Add more diverse training images (different angles, lighting)
- **Lower memory:** Process images in batches during training
- **Production:** Consider using a production WSGI server (Gunicorn, uWSGI)

## License

This project is part of a graduation project.

## Author

Part of a graduation project

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) for face detection
- [InsightFace](https://github.com/deepinsight/insightface) for face recognition
- [Flask](https://flask.palletsprojects.com/) for web framework

---

**Developed with:** Python, Flask, YOLOv5, InsightFace, Jupyter Notebook
