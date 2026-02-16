"""
Flask REST API for Vehicle Damage Cost Prediction
This API can be called from ASP.NET C# application
"""
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# --- Configuration ---
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from ASP.NET

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Global Model and Mapping ---
MODEL_PATH = 'car_damage_cost_predictor_v1.h5'
MODEL = None

# Cost Mapping - Sri Lankan Rupees (LKR)
# Converted from USD using approximate rate: 1 USD = 300 LKR
CLASS_LABELS = ['Bumper scratch', 'Door scratch', 'Bumper dent', 'Door dent', 
                'Broken headlamp', 'Broken tail lamp', 'Glass shatter', 'Unknown']
COST_MAP = {
    'Bumper scratch': 60000,      # ~$200 = 60,000 LKR
    'Door scratch': 75000,        # ~$250 = 75,000 LKR
    'Bumper dent': 150000,        # ~$500 = 150,000 LKR
    'Door dent': 210000,          # ~$700 = 210,000 LKR
    'Broken headlamp': 240000,    # ~$800 = 240,000 LKR
    'Broken tail lamp': 270000,   # ~$900 = 270,000 LKR
    'Glass shatter': 360000,      # ~$1200 = 360,000 LKR
    'Unknown': 30000              # ~$100 = 30,000 LKR
}

# --- Utility Functions ---
def load_ml_model():
    """Load the model once at startup"""
    global MODEL
    try:
        MODEL = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        MODEL = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_from_image(img):
    """
    Predict damage class and cost from PIL Image
    """
    # Resize to model input size
    img = img.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = MODEL.predict(img_array)[0]
    
    # Get predicted class and cost
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index])
    predicted_class = CLASS_LABELS[predicted_index]
    estimated_cost = COST_MAP.get(predicted_class, 0)
    
    return {
        'predicted_class': predicted_class,
        'estimated_cost': estimated_cost,
        'confidence': round(confidence * 100, 2),
        'all_predictions': {CLASS_LABELS[i]: round(float(predictions[i]) * 100, 2) 
                           for i in range(len(CLASS_LABELS))}
    }

# --- API Routes ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict_file():
    """
    Predict from uploaded file
    Accepts multipart/form-data with 'file' field
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
    
    try:
        # Read image directly from file stream
        img = Image.open(file.stream).convert('RGB')
        
        # Get prediction
        result = predict_from_image(img)
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict-base64', methods=['POST'])
def predict_base64():
    """
    Predict from base64 encoded image
    Accepts JSON: {"image": "base64_string"}
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        
        # Remove data URI prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode and open image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        
        # Get prediction
        result = predict_from_image(img)
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict-url', methods=['POST'])
def predict_url():
    """
    Predict from image URL
    Accepts JSON: {"url": "image_url"}
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        import requests
        from io import BytesIO
        
        # Download image
        response = requests.get(data['url'], timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Get prediction
        result = predict_from_image(img)
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# --- Main ---
if __name__ == '__main__':
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load model
    load_ml_model()
    
    # Start server
    # Use host='0.0.0.0' to allow external connections
    app.run(host='0.0.0.0', port=5000, debug=False)
