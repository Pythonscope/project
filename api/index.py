from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
from well_log_model import ProWellLogInterpreter

app = Flask(__name__)
CORS(app)
interpreter = ProWellLogInterpreter()

def ok(data):  return jsonify({"ok": True , "data": data})
def err(msg):  return jsonify({"ok": False, "error": msg}), 400

@app.get("/")
def alive(): return "Well-Log AI API is running on Vercel!"

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f: return err("No file")
    interpreter.preprocess_data(f)
    interpreter.generate_targets()
    return ok("CSV loaded & targets generated")

@app.post("/train")
def train():
    try:
        interpreter.train_models()          # <10 s
        return ok(interpreter.metrics)
    except Exception as e:
        return err(str(e))

@app.get("/recommend")
def recommend():
    return ok(interpreter.generate_recommendations())

@app.get("/plot")
def plot_png():
    fig = interpreter.make_plot()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# ‼️ DO NOT call app.run() – Vercel injects its own server
