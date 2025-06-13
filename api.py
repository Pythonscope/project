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
        
        return ok({
            'message': 'Enhanced AI models trained successfully',
            'metrics': interpreter.metrics,
            'model_type': 'Enhanced Ensemble Models'
        })
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return err(f"Training failed: {str(e)}", 500)

# 3 System status
@app.route('/status', methods=['GET'])
def status():
    try:
        status_info = {
            'data_loaded': interpreter.data is not None,
            'models_trained': interpreter.is_trained,
            'records_count': len(interpreter.data) if interpreter.data is not None else 0,
            'available_features': interpreter.feature_columns + interpreter.extra_features,
            'model_metrics': interpreter.metrics if interpreter.is_trained else None
        }
        return ok(status_info)
        
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        return err(f"Status check failed: {str(e)}", 500)

# 4 Feature importance
@app.route('/importance', methods=['GET'])
def importance():
    try:
        if not interpreter.is_trained:
            return err('Models not trained yet. Please train models first.')
        
        # Get feature importance from ensemble model
        if hasattr(interpreter.lithology_model, 'feature_importances_'):
            feats = interpreter.feature_columns + interpreter.extra_features
            available_feats = [f for f in feats if f in interpreter.data.columns]
            imps = interpreter.lithology_model.feature_importances_
            
            importance_dict = {f: float(v) for f, v in zip(available_feats, imps)}
            return ok(importance_dict)
        else:
            return err('Feature importance not available for ensemble models')
            
    except Exception as e:
        logger.error(f"Importance error: {str(e)}")
        return err(f"Feature importance failed: {str(e)}", 500)

# 5 Enhanced recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        if not interpreter.is_trained:
            return err('Models not trained yet. Please train models first.')
        
        recommendations = interpreter.generate_enhanced_recommendations()
        return ok(recommendations)
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return err(f"Recommendation generation failed: {str(e)}", 500)

# 6 Enhanced plot
@app.route('/plot', methods=['GET'])
def plot():
    try:
        if interpreter.data is None:
            return err('No data loaded. Please upload a CSV file first.')
        
        fig = interpreter.make_plot()
        
        # Create buffer for image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Clean up matplotlib figure
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return send_file(buf, mimetype='image/png', 
                        as_attachment=False, 
                        download_name='well_log_plot.png')
        
    except Exception as e:
        logger.error(f"Plot error: {str(e)}")
        return err(f"Plot generation failed: {str(e)}", 500)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return err('Endpoint not found', 404)

@app.errorhandler(500)
def internal_error(error):
    return err('Internal server error', 500)

# For deployment
if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
else:
    # Production server (Gunicorn)
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
