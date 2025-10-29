"""
Email Phishing Detector - Flask Web Application

A machine learning-powered web application that detects phishing emails
using neural networks and natural language processing.

Author: Student Project
Date: 2025
Technologies: Flask, TensorFlow, NLTK
Purpose: Educational demonstration of ML in cybersecurity
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import os
from huggingface_hub import hf_hub_download

# Custom InputLayer class to handle compatibility issues
class CompatibleInputLayer(tf.keras.layers.InputLayer):
    """Custom InputLayer that handles batch_shape parameter for compatibility"""
    def __init__(self, *args, **kwargs):
        # Handle the batch_shape parameter that causes issues in newer TensorFlow versions
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

# Import NLTK with error handling for deployment environments
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError as e:
    print(f"NLTK import failed: {e}")
    NLTK_AVAILABLE = False

def download_nltk_data():
    """Download required NLTK data with error handling"""
    if not NLTK_AVAILABLE:
        print("NLTK not available, using fallback stopwords")
        return False
        
    required_data = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt')
    ]
    
    for data_path, data_name in required_data:
        try:
            nltk.data.find(data_path)
            print(f"✓ NLTK {data_name} already available")
        except LookupError:
            print(f"Downloading NLTK {data_name}...")
            try:
                nltk.download(data_name, quiet=True)
                print(f"✓ NLTK {data_name} downloaded successfully")
            except Exception as e:
                print(f"Error downloading NLTK {data_name}: {e}")
                return False
        except Exception as e:
            print(f"Error checking NLTK {data_name}: {e}")
            return False
    return True

# Download required NLTK data
nltk_success = download_nltk_data()

# Import stopwords with fallback
if nltk_success and NLTK_AVAILABLE:
    try:
        from nltk.corpus import stopwords
        STOPWORDS_AVAILABLE = True
    except Exception as e:
        print(f"Error importing NLTK stopwords: {e}")
        STOPWORDS_AVAILABLE = False
else:
    STOPWORDS_AVAILABLE = False

app = Flask(__name__)

# Configuration
MAX_LENGTH = 200
MODEL_FILE = "phishing_detector_model.keras"
TOKENIZER_FILE = "tokenizer.pickle"

# Hugging Face configuration
HF_REPO_ID = "anisltr/phishing_detector_model"
HF_MODEL_FILENAME = "phishing_detector_model.keras"

# Global variables for model and tokenizer
model = None
tokenizer = None

# Initialize stopwords with fallback
if STOPWORDS_AVAILABLE:
    try:
        stop_words = set(stopwords.words('english'))
        print("✓ NLTK stopwords loaded successfully")
    except Exception as e:
        print(f"Error loading NLTK stopwords: {e}")
        stop_words = set()
else:
    # Fallback stopwords list for deployment environments without NLTK
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once'
    }
    print("✓ Using fallback stopwords (NLTK not available)")

def load_ai_model():
    """Load the AI model and tokenizer"""
    global model, tokenizer
    
    print("Loading AI model and tokenizer...")
    
    # Load model
    print("Loading model...")
    try:
        # Check if model exists locally
        if not os.path.exists(MODEL_FILE):
            print("Model not found locally. Downloading from Hugging Face...")
            try:
                # Download model from Hugging Face
                downloaded_model_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=HF_MODEL_FILENAME,
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
                print(f"Model downloaded successfully to: {downloaded_model_path}")
            except Exception as download_error:
                print(f"Error downloading model from Hugging Face: {download_error}")
                print("Please ensure the model is uploaded to Hugging Face and the repo_id is correct.")
                return False
        
        # Try loading with different compatibility options
        try:
            # First try: Standard loading
            model = tf.keras.models.load_model(MODEL_FILE)
            print("Model loaded successfully!")
        except Exception as load_error:
            print(f"Standard model loading failed: {load_error}")
            print("Trying compatibility mode...")
            
            try:
                # Second try: Load with compile=False to avoid optimizer issues
                model = tf.keras.models.load_model(MODEL_FILE, compile=False)
                
                # Recompile the model with current TensorFlow version
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print("Model loaded successfully in compatibility mode!")
                
            except Exception as compat_error:
                print(f"Compatibility mode loading failed: {compat_error}")
                print("Trying custom object loading...")
                
                try:
                    # Third try: Load with custom objects to handle version differences
                    custom_objects = {
                        'InputLayer': CompatibleInputLayer,
                    }
                    model = tf.keras.models.load_model(
                        MODEL_FILE, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    
                    # Recompile the model
                    model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    print("Model loaded successfully with custom objects!")
                    
                except Exception as custom_error:
                    print(f"Custom object loading failed: {custom_error}")
                    raise custom_error
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        with open(TOKENIZER_FILE, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Tokenizer file '{TOKENIZER_FILE}' not found.")
        return False
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return False
    
    print("✓ AI system ready!")
    return True

def clean_text(text):
    """Clean and preprocess text for prediction"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_phishing(email_text):
    """
    Predict if email text is phishing or safe
    Returns: Dictionary with prediction results
    """
    if not model or not tokenizer:
        return {"error": "AI model not loaded"}
    
    try:
        # Clean the input text
        cleaned = clean_text(email_text)
        
        # Tokenize and pad the cleaned text
        sequences = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
        
        # Make prediction
        prediction_score = model.predict(padded)[0][0]
        
        # Interpret the score
        label = 1 if prediction_score > 0.5 else 0
        confidence = float(prediction_score) if label == 1 else float(1 - prediction_score)
        
        return {
            'label': label,
            'score': float(prediction_score),
            'confidence': confidence * 100,
            'prediction': 'Phishing' if label == 1 else 'Safe',
            'cleaned_text': cleaned
        }
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for email prediction"""
    try:
        data = request.get_json()
        email_text = data.get('email_text', '').strip()
        
        if not email_text:
            return jsonify({"error": "Please provide email text"}), 400
        
        # Make prediction
        result = predict_phishing(email_text)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

if __name__ == '__main__':
    # Load AI model on startup
    if load_ai_model():
        print("Starting Flask web server...")
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Failed to load AI model. Exiting...")