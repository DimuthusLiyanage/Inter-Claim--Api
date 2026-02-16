import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# --- Configuration ---
app = Flask(__name__)
# Set the upload folder inside the static directory
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Global Model and Mapping ---
# Load your trained Keras model globally on server start
# Adjust path as necessary
MODEL_PATH = 'car_damage_cost_predictor_v1.h5'
MODEL = load_model(MODEL_PATH) 

# Cost Mapping (Must match the 8 classes your model was trained on)
# NOTE: The order of class_labels MUST match the index order from your training generator!
CLASS_LABELS = ['Bumper scratch', 'Door scratch', 'Bumper dent', 'Door dent', 
                'Broken headlamp', 'Broken tail lamp', 'Glass shatter', 'Unknown']
COST_MAP = {
    'Bumper scratch': 200,
    'Door scratch': 250,
    'Bumper dent': 500,
    'Door dent': 700,
    'Broken headlamp': 800,
    'Broken tail lamp': 900,
    'Glass shatter': 1200,
    'Unknown': 100
}

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(img_path):
    # 1. Load the image and resize to 224x224 (Model Input Size)
    img = load_img(img_path, target_size=(224, 224))
    
    # 2. Convert to Array and Preprocess
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize (Must match training preprocessing)

    # 3. Make Prediction
    predictions = MODEL.predict(img_array)[0]
    
    # 4. Convert output to final cost/class
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_LABELS[predicted_index]
    estimated_cost = COST_MAP.get(predicted_class, 'N/A')
    
    return predicted_class, estimated_cost

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # Render the initial upload form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        # Securely save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run the model prediction
        predicted_class, estimated_cost = model_predict(filepath)
        
        # Prepare results for the HTML template
        result = {
            'filepath': filepath,
            'class': predicted_class,
            'cost': f"${estimated_cost}.00" if isinstance(estimated_cost, int) else estimated_cost
        }
        
        # Pass the results back to the template
        return render_template('index.html', result=result)
    
    return 'Invalid file type', 400

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Start the Flask development server
    app.run(debug=True)