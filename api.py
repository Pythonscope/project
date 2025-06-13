# ---------- Flask REST API ---------------------------------
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import pandas as pd
import os
import logging
from well_log_model import EnhancedWellLogInterpreter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Use enhanced interpreter
interpreter = EnhancedWellLogInterpreter()

# ------------ helpers --------------------------------------
def ok(data):
    """Return JSON with NumPy scalars converted to native types."""
    import numpy as np
    import json
    
    def native(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, dict):
            return {k: native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [native(item) for item in o]
        return o
    
    return jsonify({'ok': True, 'data': native(data)})

def err(msg, code=400):
    logger.error(f"API Error: {msg}")
    return jsonify({'ok': False, 'error': str(msg)}), code

# ------------ routes ---------------------------------------
@app.route('/')
def home():
    return "AI Well Log Interpreter API is running!"

@app.route('/health')
def health():
    """Health check endpoint for deployment platforms"""
    return ok({'status': 'healthy', 'service': 'AI Well Log Interpreter'})

# 1 Upload & preprocess CSV
@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return err('No file part in request')
        
        file = request.files['file']
        if file.filename == '':
            return err('No file selected')
        
        if not file.filename.lower().endswith('.csv'):
            return err('File must be a CSV file')
        
        # Process the file
        result = interpreter.preprocess_data(file)
        
        # Return summary information
        summary = {
            'message': 'CSV loaded & preprocessed successfully',
            'rows': len(result),
            'columns': list(result.columns),
            'features_created': len(interpreter.extra_features)
        }
        
        return ok(summary)
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return err(f"Upload failed: {str(e)}", 500)

# 2 Train all models (enhanced)
@app.route('/train', methods=['POST'])
def train():
    try:
        if interpreter.data is None:
            return err('No data loaded. Please upload a CSV file first.')
        
        # Use enhanced training method
        interpreter.train_enhanced_models()
        
        return
