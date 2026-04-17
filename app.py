import os
import uuid

import pandas as pd
from flask import (
    Flask, request, render_template, jsonify, make_response,
    send_from_directory, flash, redirect, url_for,
)
from werkzeug.utils import secure_filename

# --- legacy (dlib) pipeline --------------------------------------------------
from functions_only_save import make_face_df_save, find_face_shape
from recommender import process_rec_pics, run_recommender_face_shape

# --- v2 (mediapipe + CLIP) pipeline -----------------------------------------
from face_analysis import analyze_face, annotate
from recommender_v2 import recommend as recommend_v2


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REC_PICS_DIR = os.path.join(BASE_DIR, "data", "rec_pics")
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_url_path="")
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Legacy global feature frame (kept so /predict still works untouched)
df = pd.DataFrame(columns=[str(i) for i in range(144)] +
                  [f"A{i}" for i in range(1, 17)] +
                  ["Width", "Height", "H_W_Ratio", "Jaw_width",
                   "J_F_Ratio", "MJ_width", "MJ_J_width"])


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """V2 single-page app."""
    return render_template("v2.html")


@app.route("/legacy")
def legacy():
    return render_template("theme.html")


# ---------------------------------------------------------------------------
# V2 API
# ---------------------------------------------------------------------------

@app.route("/api/v2/upload", methods=["POST"])
def api_v2_upload():
    """Accept a multipart file upload, return a server-side filename."""
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    if f.filename == "" or not _allowed(f.filename):
        return jsonify({"error": "invalid or missing file"}), 400

    ext = f.filename.rsplit(".", 1)[1].lower()
    name = f"{uuid.uuid4().hex[:12]}.{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    return jsonify({"filename": name, "url": url_for("serve_upload", filename=name)})


@app.route("/api/v2/analyze", methods=["POST"])
def api_v2_analyze():
    """
    Body (JSON): { filename: str, hair_length?: "Short"|"Long"|"Updo", k?: int }
    Returns the full recommend_v2() payload.
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": f"upload not found: {safe}"}), 404

    result = recommend_v2(
        image_path=path,
        hair_length=data.get("hair_length"),
        k=int(data.get("k", 6)),
        output_dir=OUTPUT_DIR,
    )
    if not result["success"]:
        return jsonify(result), 422

    # Rewrite filesystem paths to servable URLs.
    if result.get("annotated_img"):
        result["annotated_img"] = "/output/" + os.path.basename(result["annotated_img"])
    for r in result["recommendations"]:
        rel = os.path.relpath(r["path"], REC_PICS_DIR).replace(os.sep, "/")
        r["url"] = "/styles/" + rel

    return jsonify(result)


# ---------------------------------------------------------------------------
# Static file helpers
# ---------------------------------------------------------------------------

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/styles/<path:filename>")
def serve_style(filename):
    return send_from_directory(REC_PICS_DIR, filename)


# ---------------------------------------------------------------------------
# Legacy V1 endpoints (unchanged behaviour)
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.json
    test_photo = "data/pics/recommendation_pics/" + data["file_name"]
    file_num = 2035
    style_df = pd.DataFrame(columns=["face_shape", "hair_length", "location", "filename", "score"])
    hair_length_input = "Updo"
    updo_input = data["person_see_up_dos"]
    if updo_input in ["n", "no", "N", "No", "NO"]:
        hair_length_input = data["person_hair_length"]
        if hair_length_input in ["short", "Short", "s", "S"]:
            hair_length_input = "Short"
        if hair_length_input in ["long", "longer", "l", "L"]:
            hair_length_input = "Long"

    make_face_df_save(test_photo, file_num, df)
    face_shape = find_face_shape(df, file_num)
    process_rec_pics(style_df)
    img_filename = run_recommender_face_shape(face_shape[0], style_df, hair_length_input)
    return jsonify({"Face Shape": face_shape[0], "img_filename": img_filename})


@app.route("/predict_user_face_shape", methods=["GET", "POST"])
def predict_user_face_shape():
    data = request.json
    test_photo = "data/pics/recommendation_pics/" + data["file_name"]
    file_num = 2035
    make_face_df_save(test_photo, file_num, df)
    face_shape = find_face_shape(df, file_num)
    return jsonify({"face_shape": face_shape[0]})


@app.route("/predict_v2", methods=["POST"])
def predict_v2_compat():
    """Backward-compat alias for the earlier /predict_v2 contract."""
    data = request.json or {}
    if "file_name" not in data:
        return jsonify({"error": "file_name is required"}), 400
    test_photo = os.path.join("data/pics/recommendation_pics", data["file_name"])
    res = analyze_face(test_photo)
    if not res.success:
        return jsonify({"error": res.error}), 422
    payload = res.to_dict()
    if data.get("annotate"):
        out = os.path.join(OUTPUT_DIR, f"annotated_{secure_filename(data['file_name'])}")
        if annotate(test_photo, out):
            payload["annotated_img"] = "/output/" + os.path.basename(out)
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Legacy upload form (kept for theme.html)
# ---------------------------------------------------------------------------

@app.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)
    if file and _allowed(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_DIR, filename))
        flash("Image successfully uploaded and displayed below")
        return render_template("index.html", filename=filename)
    flash("Allowed image types are - png, jpg, jpeg, gif")
    return redirect(request.url)


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
