from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import pandas as pd
import numpy as np
import os
import logging
from well_log_model import WellLogInterpreter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global interpreter instance
interpreter = WellLogInterpreter()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ok(data):
    """Return JSON with NumPy scalars converted to native types."""
    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    return jsonify({'ok': True, 'data': convert_numpy(data)})

def err(msg, code=400):
    logger.error(f"API Error: {msg}")
    return jsonify({'ok': False, 'error': str(msg)}), code

@app.route('/')
def home():
    return ok("AI Well Log Interpreter API is running!")

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return err('No file part in request')
        
        file = request.files['file']
        if file.filename == '':
            return err('No file selected')
        
        if not allowed_file(file.filename):
            return err('Invalid file type. Only CSV files are allowed.')
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return err(f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB')
        
        # Save and process file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Preprocess the data
        interpreter.preprocess_data(filepath)
        
        # Generate synthetic targets for demo
        interpreter.generate_targets()
        
        logger.info(f"Successfully processed file: {file.filename}")
        return ok(f"CSV loaded and preprocessed. {len(interpreter.data)} records processed.")
        
    except Exception as e:
        return err(f"Upload error: {str(e)}", 500)

@app.route('/train', methods=['POST'])
def train():
    try:
        if interpreter.data is None:
            return err("No data loaded. Please upload a CSV file first.")
        
        interpreter.train_models()
        
        logger.info("Models trained successfully")
        return ok({
            "message": "AI models trained successfully",
            "metrics": interpreter.metrics,
            "features_used": interpreter.feature_columns + interpreter.extra_features
        })
        
    except Exception as e:
        return err(f"Training error: {str(e)}", 500)

@app.route('/plot', methods=['GET'])
def plot():
    try:
        if interpreter.data is None:
            return err("No data loaded. Please upload a CSV file first.")
        
        fig = interpreter.make_plot()
        
        # Save to BytesIO buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Close the figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return send_file(buf, mimetype='image/png', as_attachment=False)
        
    except Exception as e:
        return err(f"Plotting error: {str(e)}", 500)

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        if interpreter.data is None:
            return err("No data loaded. Please upload a CSV file first.")
        
        if interpreter.lithology_model is None:
            return err("Models not trained. Please train the AI models first.")
        
        recommendations = interpreter.generate_recommendations()
        
        logger.info("Recommendations generated successfully")
        return ok(recommendations)
        
    except Exception as e:
        return err(f"Recommendation error: {str(e)}", 500)

@app.route('/importance', methods=['GET'])
def importance():
    try:
        if interpreter.lithology_model is None:
            return err("Models not trained. Please train the AI models first.")
        
        features = interpreter.feature_columns + interpreter.extra_features
        importances = interpreter.lithology_model.feature_importances_
        
        feature_importance = {
            feature: float(importance) 
            for feature, importance in zip(features, importances)
        }
        
        return ok(feature_importance)
        
    except Exception as e:
        return err(f"Feature importance error: {str(e)}", 500)

@app.route('/status', methods=['GET'])
def status():
    """Get current system status"""
    try:
        status_info = {
            "data_loaded": interpreter.data is not None,
            "models_trained": interpreter.lithology_model is not None,
            "records_count": len(interpreter.data) if interpreter.data is not None else 0,
            "available_features": interpreter.feature_columns,
            "extra_features": interpreter.extra_features
        }
        return ok(status_info)
    except Exception as e:
        return err(f"Status error: {str(e)}", 500)

@app.errorhandler(413)
def too_large(e):
    return err("File too large", 413)

@app.errorhandler(500)
def internal_error(e):
    return err("Internal server error", 500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
