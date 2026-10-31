import base64
import binascii
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024

WEB_PAGE = Path(__file__).resolve().with_name("index.html")
STATE_LOCK = threading.Lock()
HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
FULL_BODY_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

STATE = {
    "frame": None,
    "frame_id": 0,
    "people": [],
    "criteria": {
        "color": "",
        "clothing_type": "",
        "fabric": "",
        "pixels_per_cm": None,
    },
}

COLOR_NAMES_AR = {
    "black": "أسود",
    "white": "أبيض",
    "gray": "رمادي",
    "red": "أحمر",
    "orange": "برتقالي",
    "yellow": "أصفر",
    "green": "أخضر",
    "blue": "أزرق",
    "purple": "بنفسجي",
    "unknown": "غير معروف",
}
ALLOWED_COLORS = set(COLOR_NAMES_AR) - {"unknown"}


def detect_color(bgr_roi):
    if bgr_roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    median_value = float(np.median(value))
    colorful = saturation > 55

    if np.count_nonzero(colorful) < bgr_roi.shape[0] * bgr_roi.shape[1] * 0.08:
        if median_value < 55:
            return "black"
        if median_value > 205:
            return "white"
        return "gray"

    hue = float(np.median(hsv[:, :, 0][colorful]))
    if hue < 8 or hue >= 170:
        return "red"
    if hue < 20:
        return "orange"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    return "purple"


def get_people(frame, criteria):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, _ = HOG.detectMultiScale(
        gray, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    if len(boxes) == 0 and not FULL_BODY_CASCADE.empty():
        boxes = FULL_BODY_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 120)
        )

    people = []
    for box in boxes:
        x, y, width, height = (int(value) for value in box)
        top = max(0, y + int(height * 0.2))
        bottom = min(frame.shape[0], y + int(height * 0.72))
        left = max(0, x + int(width * 0.18))
        right = min(frame.shape[1], x + int(width * 0.82))
        color = detect_color(frame[top:bottom, left:right])
        match = not criteria["color"] or color == criteria["color"]
        scale = criteria["pixels_per_cm"]
        height_cm = round(height / scale, 1) if scale else None
        people.append({
            "bbox": [x, y, width, height],
            "height_px": height,
            "height_cm": height_cm,
            "color": color,
            "color_ar": COLOR_NAMES_AR.get(color, "غير معروف"),
            "matches": match,
        })

    return people


def process_frame(frame):
    with STATE_LOCK:
        criteria = dict(STATE["criteria"])
    people = get_people(frame, criteria)
    for person in people:
        x, y, width, height = person["bbox"]
        color = (0, 220, 0) if person["matches"] else (0, 120, 255)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
        height_text = "{} px".format(person["height_px"])
        if person["height_cm"] is not None:
            height_text = "{:.1f} cm".format(person["height_cm"])
        label = "{} | {}".format(person["color"], height_text)
        cv2.putText(
            frame, label, (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, 2, cv2.LINE_AA
        )
    if not people:
        cv2.putText(
            frame, "No person detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2, cv2.LINE_AA
        )

    height, width = frame.shape[:2]
    if width > 900:
        new_height = int(height * 900 / width)
        frame = cv2.resize(frame, (900, new_height))
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise ValueError("Could not encode processed camera frame.")
    return encoded.tobytes(), people


@app.get("/")
def index():
    return send_file(WEB_PAGE)


@app.post("/upload")
def upload_frame():
    payload = request.get_json(silent=True)
    frame_data = payload.get("frame") if isinstance(payload, dict) else None
    if not isinstance(frame_data, str) or not frame_data:
        return jsonify({"error": "A camera frame is required."}), 400

    encoded_frame = frame_data.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded_frame, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"error": "The camera frame is not valid base64."}), 400
    if len(image_bytes) > 4 * 1024 * 1024:
        return jsonify({"error": "The camera frame is too large."}), 413

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "The uploaded camera frame is not a valid image."}), 400

    try:
        processed_frame, people = process_frame(frame)
    except ValueError as error:
        return jsonify({"error": str(error)}), 500

    with STATE_LOCK:
        STATE["frame"] = processed_frame
        STATE["frame_id"] += 1
        STATE["people"] = people

    return jsonify({"person_count": len(people)})


@app.get("/api/status")
def status():
    with STATE_LOCK:
        people = [dict(person) for person in STATE["people"]]
        frame_id = STATE["frame_id"]
        has_frame = STATE["frame"] is not None
    return jsonify({
        "has_frame": has_frame,
        "frame_id": frame_id,
        "people": people,
    })


@app.get("/api/latest.png")
def latest_frame():
    with STATE_LOCK:
        frame = STATE["frame"]
    if frame is None:
        return jsonify({"error": "No phone camera frame has arrived yet."}), 404
    from flask import Response

    return Response(frame, mimetype="image/png")


@app.post("/api/criteria")
def update_criteria():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Search criteria must be a JSON object."}), 400

    color = payload.get("color", "")
    if color not in ALLOWED_COLORS and color != "":
        return jsonify({"error": "Unsupported clothing color."}), 400

    scale = payload.get("pixels_per_cm")
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return jsonify({"error": "Calibration must be a number."}), 400
        if not 0.1 <= scale <= 100:
            return jsonify({"error": "Calibration must be between 0.1 and 100."}), 400

    with STATE_LOCK:
        criteria = {
            "color": color,
            "clothing_type": str(payload.get("clothing_type", ""))[:100],
            "fabric": str(payload.get("fabric", ""))[:100],
            "pixels_per_cm": scale,
        }
        STATE["criteria"] = criteria
        people = []
        for existing_person in STATE["people"]:
            person = dict(existing_person)
            person["matches"] = not color or person["color"] == color
            person["height_cm"] = (
                round(person["height_px"] / scale, 1) if scale else None
            )
            people.append(person)
        STATE["people"] = people

    matches = [person for person in people if person["matches"]]
    return jsonify({
        "people": people,
        "matches": matches,
        "criteria": criteria,
    })


def start_server():
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True,
        ssl_context="adhoc",
    )


if __name__ == "__main__":
    start_server()
