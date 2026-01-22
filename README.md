# Face Recognition System

> **Note:** This project is part of a graduation project.

## 🎥 Demo Video

### Option 1: Watch on GitHub
[![Face Recognition Demo](demo_video.mp4)](demo_video.mp4)

*Click the image above to watch the demo video (demo_video.mp4)*

### Option 2: Upload to YouTube
If you prefer to host the video on YouTube, replace `YOUR_VIDEO_ID` in the link below with your YouTube video ID:

```markdown
[![Face Recognition Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)
```

**Note:** GitHub supports video playback for `.mp4` files. If you want better compatibility, convert the video to MP4 format or upload it to YouTube/Vimeo.

## Overview

A complete face recognition system using Flask, YOLOv5, and InsightFace. This project provides a simple and professional web interface for testing face recognition, along with a complete training pipeline in Jupyter Notebook.

## Features

- ✅ Modern and user-friendly web interface
- ✅ Real-time face recognition
- ✅ Adjustable recognition threshold
- ✅ System statistics
- ✅ Drag and drop support
- ✅ Responsive design
- ✅ Complete training pipeline (Jupyter Notebook)
- ✅ Sample data and models included

## Project Structure

```
insightface_github_complete/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── Face_Recognition_Pipeline.ipynb # Complete training pipeline
├── demo_video.mp4                  # Demo video
├── static/
│   ├── script.js                  # Frontend JavaScript
│   └── style.css                  # Frontend styles
├── templates/
│   └── index.html                 # Main HTML template
├── Face_detection/                 # Models and embeddings
│   ├── weights/
│   │   └── best.pt                # YOLOv5 weights (sample)
│   ├── embeddings_raw.pkl         # Face embeddings database (sample)
│   └── insightface/               # InsightFace models
├── data/                          # Sample training data
│   ├── sample/                    # Original sample images
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

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd insightface_github_complete
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

4. **Install Jupyter Notebook (for training)**
   ```bash
   pip install jupyter notebook
   ```

## Usage

### Web Application

1. **Run the Flask application**
   ```bash
   python app.py
   ```

2. **Open browser**
   - Navigate to: http://127.0.0.1:5000

3. **Use the interface**
   - Upload an image containing a face
   - Adjust threshold as needed (default: 0.45)
   - Click "Recognize Face"
   - View recognition results and confidence scores

### Training Pipeline

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Run the training pipeline**
   - Open `Face_Recognition_Pipeline.ipynb`
   - Follow the 14-step guide to:
     - Train YOLOv5 model for face detection
     - Extract face embeddings using InsightFace
     - Build the recognition database
     - Test the system

## Sample Data

This repository includes sample data for testing:
- **Sample images** in `data/sample/` folder (organized by person name)
- **Cropped faces** in `data/cropped_faces/` folder (processed images)
- **Pre-trained model** (`best.pt`) in `Face_detection/weights/`
- **Sample embeddings** (`embeddings_raw.pkl`) in `Face_detection/`

> **Note:** The included models and data are samples for demonstration. For production use, train your own models with your dataset.

## Technologies Used

- **Backend:** Flask, Python
- **Face Detection:** YOLOv5
- **Face Recognition:** InsightFace
- **Frontend:** HTML, CSS, JavaScript
- **ML Libraries:** scikit-learn, NumPy, OpenCV
- **Notebook:** Jupyter Notebook

## Important Notes

1. **Sample Models:** The included models (`best.pt` and `embeddings_raw.pkl`) are samples for testing. For production, train your own models with your dataset.

2. **Environment:** Make sure to install Visual C++ Redistributable 2019 before running the application on Windows.

3. **Security:** This application is intended for local/development use. Do not use it in production without proper security measures.

4. **Training:** Use the provided Jupyter Notebook to train models on your own data.

## Troubleshooting

### Issue: "onnxruntime DLL error"
**Solution:**
1. Install Visual C++ Redistributable 2019
2. Restart Terminal/IDE
3. Reinstall onnxruntime:
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime
   ```

### Issue: "No module named 'yolov5'"
**Solution:**
```bash
pip install yolov5
```

### Issue: "No face detected"
**Solution:** Make sure the image contains a clear and visible face.

### Issue: Models not loading
**Solution:** 
- Check that `best.pt` exists in `Face_detection/weights/`
- Check that `embeddings_raw.pkl` exists in `Face_detection/`
- Run the training pipeline to generate your own models

## Training Your Own Models

1. **Prepare your dataset**
   - Organize images in folders by person name
   - Place in `data/` directory

2. **Run the training pipeline**
   - Open `Face_Recognition_Pipeline.ipynb`
   - Follow all 14 steps
   - The notebook will generate:
     - Trained YOLOv5 model (`best.pt`)
     - Face embeddings database (`embeddings_raw.pkl`)

3. **Test your models**
   - Run `app.py` to test with the web interface
   - Upload test images to verify recognition accuracy

## License

This project is part of a graduation project.

## Author

Part of a graduation project

---

**Developed with:** Python, Flask, YOLOv5, InsightFace, Jupyter Notebook
# insightface-flask-app
