import json
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from inference import preview_image, style_transfer

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
GALLERY_FOLDER = BASE_DIR / "static" / "gallery"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MANIFEST_PATH = GALLERY_FOLDER / "manifest.json"
MAX_SIZE = 512

for folder in (UPLOAD_FOLDER, RESULT_FOLDER, GALLERY_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def static_url(path: Path) -> str:
    return url_for("static", filename=path.relative_to(BASE_DIR / "static").as_posix())


def find_upload(uid: str, prefix: str):
    matches = sorted(UPLOAD_FOLDER.glob(f"{uid}_{prefix}_*"))
    return matches[0] if matches else None


def list_results(uid: str):
    results = []
    for p in RESULT_FOLDER.glob(f"{uid}_result_*.jpg"):
        alpha = int(p.stem.rsplit("_", 1)[-1]) / 100
        results.append({"alpha": alpha, "path": p})
    results.sort(key=lambda r: r["path"].stat().st_mtime, reverse=True)
    return results


def ensure_preview(uid: str, content_path: Path) -> Path:
    before = RESULT_FOLDER / f"{uid}_before.jpg"
    if not before.exists():
        preview_image(content_path, MAX_SIZE).save(before)
    return before


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stylize", methods=["POST"])
def stylize():
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    content_file = request.files.get("content_image")
    style_file = request.files.get("style_image")
    alpha = min(max(float(request.form.get("alpha", 0.7)), 0.0), 1.0)

    def bad(msg, code=400):
        if is_ajax:
            return jsonify({"error": msg}), code
        return render_template("index.html", error=msg), code

    if not content_file or not style_file or not content_file.filename or not style_file.filename:
        return bad("Please upload both images.")
    if not allowed_file(content_file.filename) or not allowed_file(style_file.filename):
        return bad("Please upload PNG, JPG, JPEG, BMP, or WEBP images.")

    uid = uuid.uuid4().hex
    content_name = f"{uid}_content_{secure_filename(content_file.filename)}"
    style_name = f"{uid}_style_{secure_filename(style_file.filename)}"
    content_path = UPLOAD_FOLDER / content_name
    style_path = UPLOAD_FOLDER / style_name
    content_file.save(content_path)
    style_file.save(style_path)

    alpha_tag = str(int(round(alpha * 100))).zfill(2)
    result_path = RESULT_FOLDER / f"{uid}_result_{alpha_tag}.jpg"
    try:
        result_img = style_transfer(content_path, style_path, alpha=alpha, max_size=MAX_SIZE)
        result_img.save(result_path)
        before = ensure_preview(uid, content_path)
    except Exception as exc:
        return bad(f"Processing failed: {exc}", 500)

    if is_ajax:
        return jsonify({
            "uid": uid,
            "content_url": static_url(content_path),
            "style_url": static_url(style_path),
            "before_url": static_url(before),
            "result_url": static_url(result_path),
            "alpha": alpha,
        })
    return redirect(url_for("stylize_page", uid=uid))


@app.route("/stylize", methods=["GET"])
def stylize_page():
    uid = request.args.get("uid", "")
    if not uid:
        return redirect(url_for("index"))

    content_path = find_upload(uid, "content")
    style_path = find_upload(uid, "style")
    if not content_path or not style_path:
        return redirect(url_for("index"))

    before = ensure_preview(uid, content_path)
    results = list_results(uid)
    result = results[0] if results else None

    return render_template(
        "result.html",
        uid=uid,
        content_url=static_url(content_path),
        style_url=static_url(style_path),
        before_url=static_url(before),
        result_url=static_url(result["path"]) if result else None,
        result_alpha=result["alpha"] if result else None,
    )


@app.route("/restylize", methods=["POST"])
def restylize():
    data = request.get_json(silent=True) or request.form
    uid = data.get("uid", "")
    alpha = min(max(float(data.get("alpha", 0.7)), 0.0), 1.0)

    content_path = find_upload(uid, "content")
    style_path = find_upload(uid, "style")
    if not content_path or not style_path:
        return jsonify({"error": "Session not found."}), 404

    ensure_preview(uid, content_path)
    alpha_tag = str(int(round(alpha * 100))).zfill(2)
    result_path = RESULT_FOLDER / f"{uid}_result_{alpha_tag}.jpg"
    try:
        result_img = style_transfer(content_path, style_path, alpha=alpha, max_size=MAX_SIZE)
        result_img.save(result_path)
    except Exception as exc:
        return jsonify({"error": f"Processing failed: {exc}"}), 500

    return jsonify({"result_url": static_url(result_path), "alpha": alpha})


@app.route("/api/status")
def api_status():
    uid = request.args.get("uid", "")
    results = list_results(uid)
    if results:
        return jsonify({"ready": True, "result_url": static_url(results[0]["path"])})
    return jsonify({"ready": False})


@app.route("/api/samples")
def api_samples():
    if MANIFEST_PATH.exists():
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for item in data:
            item["content_url"] = url_for("static", filename=f"gallery/{item['content']}")
            item["style_url"] = url_for("static", filename=f"gallery/{item['style']}")
            item["result_url"] = url_for("static", filename=f"gallery/{item['result']}")
        return jsonify(data)
    return jsonify([])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
