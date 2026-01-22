"""
Flask Application for Face Recognition System
Simple and professional web interface for testing face recognition
"""

from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
import base64
import sys

# Try to import required libraries
INSIGHTFACE_AVAILABLE = False
ONNXRUNTIME_AVAILABLE = False
YOLOV5_AVAILABLE = False

# First check onnxruntime
try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
    print("onnxruntime imported successfully")
except Exception as e:
    ONNXRUNTIME_AVAILABLE = False
    print(f"\nWARNING: Cannot import onnxruntime: {e}")
    
    # Try to install older version of onnxruntime as alternative
    print("\nAttempting to install older version of onnxruntime (1.15.1) as alternative...")
    try:
        import subprocess
        print("Uninstalling current onnxruntime...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "-y"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
        
        print("Installing onnxruntime==1.15.1...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime==1.15.1", "--quiet"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Try importing again
        try:
            import onnxruntime
            ONNXRUNTIME_AVAILABLE = True
            print("Successfully installed and imported onnxruntime 1.15.1!")
        except Exception as import_error:
            print(f"Failed to import onnxruntime 1.15.1: {import_error}")
            print("\nCRITICAL: You MUST install Visual C++ Redistributable 2019")
            print("Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("After installation, RESTART your terminal/IDE and run this script again.")
    except Exception as install_error:
        print(f"Could not automatically install onnxruntime: {install_error}")
        print("\nCRITICAL: You MUST install Visual C++ Redistributable 2019")
        print("Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("After installation, RESTART your terminal/IDE and run:")
        print("  pip uninstall onnxruntime")
        print("  pip install onnxruntime")

# Then try insightface
if ONNXRUNTIME_AVAILABLE:
    try:
        import insightface
        INSIGHTFACE_AVAILABLE = True
        print("insightface imported successfully")
    except ImportError as e:
        INSIGHTFACE_AVAILABLE = False
        print("\n" + "=" * 60)
        print("ERROR: Cannot import InsightFace")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nSOLUTION:")
        print("1. Install Visual C++ Redistributable 2019:")
        print("   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("2. After installation, RESTART your terminal/IDE")
        print("3. Try reinstalling onnxruntime:")
        print("   pip uninstall onnxruntime")
        print("   pip install onnxruntime")
        print("4. Or try CPU-only version:")
        print("   pip uninstall onnxruntime")
        print("   pip install onnxruntime-cpu")
        print("=" * 60 + "\n")
else:
    print("\n" + "=" * 60)
    print("ERROR: Cannot import onnxruntime")
    print("=" * 60)
    print("\nSOLUTION:")
    print("1. Install Visual C++ Redistributable 2019:")
    print("   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. After installation, RESTART your terminal/IDE")
    print("3. Try reinstalling onnxruntime:")
    print("   pip uninstall onnxruntime")
    print("   pip install onnxruntime")
    print("4. Or try CPU-only version:")
    print("   pip uninstall onnxruntime")
    print("   pip install onnxruntime-cpu")
    print("=" * 60 + "\n")

# Try to import YOLOv5
try:
    import yolov5
    YOLOV5_AVAILABLE = True
    print("yolov5 imported successfully")
except ImportError as e:
    YOLOV5_AVAILABLE = False
    print(f"\nWARNING: Cannot import yolov5: {e}")
    print("\nSOLUTION:")
    print("  pip install yolov5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
yolo_model = None  # YOLOv5 for detection
face_app = None     # InsightFace for embedding only
embeddings_db = None
person_embeddings = None

# Paths configuration - Using relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "Face_detection", "weights", "best.pt")
DATA_ROOT = os.path.join(BASE_DIR, "Face_detection")
EMB_DB_PATH = os.path.join(DATA_ROOT, "embeddings_raw.pkl")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load YOLOv5 and InsightFace models"""
    global yolo_model, face_app, embeddings_db, person_embeddings
    
    # Load YOLOv5 for detection
    if not YOLOV5_AVAILABLE:
        print("\nCannot load YOLOv5: yolov5 is not available")
        print("Please install: pip install yolov5")
        return False
    
    if yolo_model is None:
        if not os.path.exists(WEIGHTS_PATH):
            print(f"\nERROR: YOLOv5 weights not found at: {WEIGHTS_PATH}")
            print("Please make sure best.pt exists in Face_detection/weights folder")
            return False
        
        try:
            print("Loading YOLOv5 model for face detection...")
            yolo_model = yolov5.load(WEIGHTS_PATH)
            yolo_model.conf = 0.25  # Confidence threshold
            yolo_model.iou = 0.45   # IoU threshold
            print("YOLOv5 loaded successfully")
        except Exception as e:
            print(f"\nError loading YOLOv5: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Load InsightFace for embedding only
    if not INSIGHTFACE_AVAILABLE:
        print("\nCannot load InsightFace: insightface is not available")
        print("Please fix the onnxruntime DLL issue first (see error message above)")
        return False
    
    if face_app is None:
        try:
            print("Loading InsightFace model for embedding...")
            face_app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace loaded successfully")
        except Exception as e:
            print(f"\nError initializing InsightFace: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Visual C++ Redistributable 2019 is installed")
            print("2. Try: pip uninstall onnxruntime && pip install onnxruntime")
            print("3. Restart your terminal/IDE after installing Visual C++ Redistributable")
            return False
    
    # Load embeddings database
    if os.path.exists(EMB_DB_PATH):
        with open(EMB_DB_PATH, 'rb') as f:
            embeddings_db = pickle.load(f)
        
        # Pre-compute average embeddings for each person
        person_embeddings = {}
        for person_name, embeddings_list in embeddings_db.items():
            embeddings_array = np.array([e['embedding'] for e in embeddings_list])
            avg_embedding = np.mean(embeddings_array, axis=0)
            person_embeddings[person_name] = avg_embedding
        
        print(f"Loaded {len(person_embeddings)} known persons")
        return True
    else:
        print(f"\nEmbeddings database not found at: {EMB_DB_PATH}")
        print("Please run the face recognition pipeline (Steps 1-14) first to generate embeddings")
        return False


def recognize_face(image_path, threshold=0.45):
    """Recognize face in an image using YOLOv5 for detection and InsightFace for embedding"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Could not read image"
        
        # Step 1: Detect faces using YOLOv5
        results = yolo_model(image_path, size=640)
        
        # Get detections
        try:
            detections_df = results.pandas().xyxy[0]
            if detections_df.empty or len(detections_df) == 0:
                return None, "No face detected in image"
            
            # Filter only faces (class 0)
            face_detections = detections_df[detections_df['class'] == 0].copy()
            if len(face_detections) == 0:
                return None, "No face detected in image"
        except (AttributeError, IndexError, KeyError) as e:
            return None, f"Error processing YOLOv5 detections: {str(e)}"
        
        # Step 2: Crop the first detected face
        h, w = img.shape[:2]
        first_face = face_detections.iloc[0]
        x1, y1, x2, y2 = int(first_face['xmin']), int(first_face['ymin']), int(first_face['xmax']), int(first_face['ymax'])
        
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop face
        face_crop = img[y1:y2, x1:x2]
        
        if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
            return None, "Face crop too small"
        
        # Step 3: Get embedding using InsightFace (on cropped face)
        faces_detected = face_app.get(face_crop)
        if len(faces_detected) == 0:
            return None, "Could not extract embedding from cropped face"
        
        # Get embedding from first face
        query_embedding = faces_detected[0].embedding
        
        # Step 4: Compare with all known persons
        best_match = None
        best_similarity = 0
        
        for person_name, person_embedding in person_embeddings.items():
            similarity = cosine_similarity([query_embedding], [person_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        if best_similarity >= threshold:
            return {
                'person_name': best_match,
                'confidence': float(best_similarity),
                'threshold': threshold,
                'status': 'recognized'
            }, None
        else:
            return {
                'person_name': 'Unknown',
                'confidence': float(best_similarity),
                'threshold': threshold,
                'status': 'unknown'
            }, None
            
    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    """Handle face recognition request"""
    try:
        if not YOLOV5_AVAILABLE or yolo_model is None or not INSIGHTFACE_AVAILABLE or face_app is None:
            return jsonify({
                'error': 'Face recognition models not loaded. Please check server logs for details.',
                'details': 'Make sure yolov5, onnxruntime and insightface are installed.'
            }), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF'}), 400
        
        # Get threshold from request (default 0.45)
        threshold = float(request.form.get('threshold', 0.45))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Recognize face
        result, error = recognize_face(filepath, threshold)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        if person_embeddings is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        stats = {
            'total_persons': len(person_embeddings),
            'total_embeddings': sum(len(emb) for emb in embeddings_db.values()) if embeddings_db else 0,
            'yolo_model_loaded': yolo_model is not None,
            'insightface_model_loaded': face_app is not None
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Face Recognition Flask App")
    print("=" * 60)
    
    # Check Python environment
    import sys
    print(f"\nPython executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    if not ONNXRUNTIME_AVAILABLE:
        print("\n" + "=" * 60)
        print("ERROR: onnxruntime is not available")
        print("=" * 60)
        print("\nPlease follow these steps:")
        print("1. Make sure you're in the correct virtual environment")
        print("2. Install Visual C++ Redistributable 2019:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("3. RESTART your terminal/IDE after installing Visual C++")
        print("4. Reinstall onnxruntime:")
        print("   pip uninstall onnxruntime")
        print("   pip install onnxruntime")
        print("5. Or try CPU-only version:")
        print("   pip uninstall onnxruntime")
        print("   pip install onnxruntime-cpu")
        print("=" * 60)
        sys.exit(1)
    
    if not YOLOV5_AVAILABLE:
        print("\nCannot start server: YOLOv5 is not available")
        print("\nPlease install:")
        print("  pip install yolov5")
        sys.exit(1)
    
    if not INSIGHTFACE_AVAILABLE:
        print("\nCannot start server: InsightFace is not available")
        print("\nPlease follow these steps:")
        print("1. Install Visual C++ Redistributable 2019:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("2. RESTART your terminal/IDE")
        print("3. Reinstall onnxruntime:")
        print("   pip uninstall onnxruntime")
        print("   pip install onnxruntime")
        print("4. Or try CPU-only version:")
        print("   pip uninstall onnxruntime")
        print("   pip install onnxruntime-cpu")
        print("5. Run this script again")
        sys.exit(1)
    
    # Load models on startup
    if load_models():
        print("\n" + "=" * 60)
        print("Models loaded successfully!")
        print("=" * 60)
        print(f"Known persons: {len(person_embeddings) if person_embeddings else 0}")
        print(f"Total embeddings: {sum(len(emb) for emb in embeddings_db.values()) if embeddings_db else 0}")
        print("\nStarting Flask server...")
        print("Open http://127.0.0.1:5000 in your browser")
        print("=" * 60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "=" * 60)
        print("Failed to load models")
        print("=" * 60)
        print("Please check:")
        print(f"1. YOLOv5 weights exist at: {WEIGHTS_PATH}")
        print(f"2. Embeddings database exists at: {EMB_DB_PATH}")
        print("3. Visual C++ Redistributable 2019 is installed")
        print("4. All dependencies are installed (yolov5, insightface, onnxruntime)")
        print("5. Run the face recognition pipeline (Steps 1-14) if embeddings don't exist")
        print("=" * 60)
        sys.exit(1)
