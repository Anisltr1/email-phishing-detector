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
import nltk
import os
from huggingface_hub import hf_hub_download

def download_nltk_data():
    """Download required NLTK data"""
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

# Download required NLTK data
download_nltk_data()

from nltk.corpus import stopwords

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

# Initialize stopwords after ensuring download
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Error loading stopwords: {e}")
    stop_words = set()  # Fallback to empty set

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
        
        model = tf.keras.models.load_model(MODEL_FILE)
        print("Model loaded successfully!")
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