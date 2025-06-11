# ---------- Flask REST API ---------------------------------
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io, pandas as pd
from well_log_model import WellLogInterpreter

app = Flask(__name__)
CORS(app)                                  # allow JS calls from WordPress
interpreter = WellLogInterpreter()         # single in-memory model instance


# ------------ helpers --------------------------------------
def ok(data):
    """Return JSON with NumPy scalars converted to native types."""
    import numpy as np, json
    def native(o):
        if isinstance(o, (np.generic,)):          # np.float32 / int64 …
            return o.item()
        if isinstance(o, (dict, list, tuple)):
            return json.loads(json.dumps(o, default=native))
        return o
    return jsonify({'ok': True,  'data': native(data)})

def err(msg, code=400):
    return jsonify({'ok': False, 'error': msg}), code


# ------------ routes ---------------------------------------
@app.route('/')
def home():
    return "Well-Log AI API is running!"

# 1 Upload & preprocess CSV
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return err('No file part')
    f = request.files['file']
    if f.filename == '':
        return err('No file selected')
    try:
        interpreter.preprocess_data(f)
        return ok('CSV loaded & pre-processed')
    except Exception as e:
        return err(str(e), 500)

# 2 Generate synthetic targets (demo only)
@app.route('/targets', methods=['POST'])
def targets():
    try:
        interpreter.generate_targets()
        return ok(interpreter.data['LITHOLOGY'].value_counts().to_dict())
    except Exception as e:
        return err(str(e), 500)

# 3 Train all models
@app.route('/train', methods=['POST'])
def train():
    try:
        interpreter.train_models()
        return ok(interpreter.metrics)
    except Exception as e:
        return err(str(e), 500)

# 4 Feature-importance dictionary
@app.route('/importance', methods=['GET'])
def importance():
    try:
        feats = interpreter.feature_columns + interpreter.extra_features
        imps  = interpreter.lithology_model.feature_importances_.round(4)
        return ok({f: float(v) for f, v in zip(feats, imps)})
    except Exception as e:
        return err(str(e), 500)

# 5 Recommendations (multi-line text)
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        return ok(interpreter.generate_recommendations())
    except Exception as e:
        return err(str(e), 500)

# 6 Professional multi-track log plot (PNG)
@app.route('/plot', methods=['GET'])
def plot():
    try:
        fig = interpreter.make_plot()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return err(str(e), 500)


if __name__ == '__main__':
    app.run(debug=False, port=5000)
