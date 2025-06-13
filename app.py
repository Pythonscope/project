from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return jsonify(ok=True, msg="AI Well Log Interpreter API is running.")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify(ok=False, error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(ok=False, error="No selected file"), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    # You can add your CSV preprocessing here
    return jsonify(ok=True, data="File uploaded and processed.")

@app.route('/train', methods=['POST'])
def train():
    # Add your AI training logic here
    # For now, just simulate a response
    return jsonify(ok=True, data="Model trained successfully.")

@app.route('/plot', methods=['GET'])
def plot():
    # Add your plotting logic here
    # For now, just return a placeholder
    return jsonify(ok=True, data="Plot generated (placeholder).")

@app.route('/recommend', methods=['GET'])
def recommend():
    # Add your recommendation logic here
    # For now, just return a placeholder
    return jsonify(ok=True, data="AI recommendations generated.")

@app.route('/download', methods=['GET'])
def download():
    # Example: serve a static report file
    report_path = os.path.join(UPLOAD_FOLDER, 'report.txt')
    if not os.path.exists(report_path):
        with open(report_path, 'w') as f:
            f.write("Sample AI Well Log Report\n")
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
