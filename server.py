import base64
import binascii
import os
import re
import threading
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024

WEB_PAGE = Path(__file__).resolve().with_name("index.html")
PROCESS_WIDTH = 480
STATE_LOCK = threading.Lock()
PROCESSOR_LOCK = threading.Lock()
FRAME_READY = threading.Event()
PROCESSOR_THREAD = None
HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.setNumThreads(1)

STATE = {
    "raw_frame": None,
    "raw_frame_id": 0,
    "frame": None,
    "frame_id": 0,
    "people": [],
    "matches": [],
    "criteria_version": 0,
    "criteria": {
        "description": "",
        "constraints": [],
        "reference_features": None,
    },
}

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "").rstrip("/")
COLOR_TERMS = {
    "black": ("اسود", "سوداء", "black"),
    "white": ("ابيض", "بيضاء", "white"),
    "red": ("احمر", "حمراء", "red"),
    "orange": ("برتقالي", "برتقالية", "orange"),
    "yellow": ("اصفر", "صفراء", "yellow"),
    "green": ("اخضر", "خضراء", "green"),
    "blue": ("ازرق", "زرقاء", "كحلي", "blue"),
    "purple": ("بنفسجي", "بنفسجية", "purple"),
    "gray": ("رمادي", "رمادية", "grey", "gray"),
    "brown": ("بني", "بنية", "brown"),
}
UPPER_CLOTHING = {
    "قميص", "بلوزه", "تيشيرت", "تي", "جاكيت", "ستره", "معطف", "كنزه",
    "بلوزة", "سترة", "كنزة", "سويتر", "shirt", "jacket", "sweater",
    "hoodie", "coat",
}
LOWER_CLOTHING = {
    "بنطلون", "سروال", "شورت", "جينز", "تنوره", "تنورة", "pants",
    "trousers", "shorts", "jeans", "skirt",
}
KNOWN_WORDS = set(UPPER_CLOTHING) | set(LOWER_CLOTHING)


@app.after_request
def add_frontend_cors_headers(response):
    origin = request.headers.get("Origin", "").rstrip("/")
    if origin and origin == FRONTEND_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def normalize_text(value):
    text = unicodedata.normalize("NFKD", value).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("ـ", "")
    return text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))


KNOWN_WORDS.update(
    normalize_text(alias) for aliases in COLOR_TERMS.values() for alias in aliases
)


def normalize_word(value):
    word = normalize_text(value)
    prefixes = ("وال", "بال", "لل", "ال", "و", "ب", "ل")
    while word not in KNOWN_WORDS:
        shortened = next(
            (
                word[len(prefix):]
                for prefix in prefixes
                if word.startswith(prefix) and len(word) > len(prefix) + 2
            ),
            None,
        )
        if shortened is None or shortened == word:
            break
        word = shortened
    return word


def parse_description(description):
    words = [
        normalize_word(word)
        for word in re.findall(r"[\w]+", normalize_text(description))
    ]
    upper_positions = [
        index for index, word in enumerate(words) if word in UPPER_CLOTHING
    ]
    lower_positions = [
        index for index, word in enumerate(words) if word in LOWER_CLOTHING
    ]
    constraints = []
    for color, aliases in COLOR_TERMS.items():
        color_words = {normalize_text(alias) for alias in aliases}
        for index, word in enumerate(words):
            if word not in color_words:
                continue
            upper_distance = min(
                (abs(index - position) for position in upper_positions),
                default=4,
            )
            lower_distance = min(
                (abs(index - position) for position in lower_positions),
                default=4,
            )
            if min(upper_distance, lower_distance) > 3:
                zone = "any"
            else:
                zone = "upper" if upper_distance <= lower_distance else "lower"
            constraints.append({"color": color, "zone": zone})
    return constraints


def detect_color(bgr_roi):
    if bgr_roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colorful = saturation > 55
    if np.count_nonzero(colorful) < bgr_roi.shape[0] * bgr_roi.shape[1] * 0.08:
        median_value = float(np.median(value))
        if median_value < 55:
            return "black"
        if median_value > 205:
            return "white"
        return "gray"

    hue = float(np.median(hsv[:, :, 0][colorful]))
    median_value = float(np.median(value[colorful]))
    if hue < 8 or hue >= 170:
        return "brown" if median_value < 115 else "red"
    if hue < 20:
        return "brown" if median_value < 135 else "orange"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    return "purple"


def detect_people(frame):
    boxes, weights = HOG.detectMultiScale(
        frame,
        hitThreshold=0,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.08,
        groupThreshold=2.0,
    )
    people = []
    frame_height, frame_width = frame.shape[:2]
    for box, weight in zip(boxes, weights):
        x, y, width, height = (int(value) for value in box)
        aspect_ratio = width / max(height, 1)
        if (
            float(weight) < 0.2
            or width < 32
            or height < 90
            or not 0.2 <= aspect_ratio <= 0.9
        ):
            continue
        x1 = max(0, x + int(width * 0.18))
        x2 = min(frame_width, x + int(width * 0.82))
        upper = frame[
            max(0, y + int(height * 0.2)):min(frame_height, y + int(height * 0.55)),
            x1:x2,
        ]
        lower = frame[
            max(0, y + int(height * 0.55)):min(frame_height, y + int(height * 0.9)),
            x1:x2,
        ]
        people.append({
            "bbox": [x, y, width, height],
            "upper_color": detect_color(upper),
            "lower_color": detect_color(lower),
        })
    return people


def appearance_descriptor(frame, bbox):
    x, y, width, height = bbox
    left = max(0, x + int(width * 0.15))
    right = min(frame.shape[1], x + int(width * 0.85))
    top = max(0, y + int(height * 0.16))
    bottom = min(frame.shape[0], y + int(height * 0.92))
    person = frame[top:bottom, left:right]
    if person.size == 0:
        return None

    split = max(1, int(person.shape[0] * 0.52))
    features = []
    for region in (person[:split], person[split:]):
        if region.size == 0:
            return None
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, alpha=1, norm_type=cv2.NORM_L1)
        features.append(histogram)
    return features


def decode_image_data(image_data):
    if not isinstance(image_data, str) or not image_data:
        raise ValueError("اختر صورة صالحة للشخص.")
    encoded = image_data.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("الصورة المرسلة غير صالحة.") from error
    if len(image_bytes) > 4 * 1024 * 1024:
        raise ValueError("حجم الصورة أكبر من الحد المسموح.")
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("تعذر قراءة الصورة. اختر صورة JPG أو PNG.")
    return frame


def reference_features(image_data):
    frame = decode_image_data(image_data)
    height, width = frame.shape[:2]
    scale = min(1.0, PROCESS_WIDTH / width, 640 / height)
    if scale < 1:
        frame = cv2.resize(
            frame, (max(1, int(width * scale)), max(1, int(height * scale)))
        )
    people = detect_people(frame)
    if people:
        person = max(people, key=lambda item: item["bbox"][2] * item["bbox"][3])
        bbox = person["bbox"]
    else:
        height, width = frame.shape[:2]
        bbox = [int(width * 0.1), int(height * 0.04), int(width * 0.8), int(height * 0.92)]
    return appearance_descriptor(frame, bbox)


def matches_description(person, constraints):
    for constraint in constraints:
        color = constraint["color"]
        zone = constraint["zone"]
        detected = (
            (person["upper_color"], person["lower_color"])
            if zone == "any"
            else (person["upper_color"],)
            if zone == "upper"
            else (person["lower_color"],)
        )
        if color not in detected:
            return False
    return True


def appearance_distance(first, second):
    if first is None or second is None:
        return 1.0
    distances = [
        cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
        for a, b in zip(first, second)
    ]
    return float(sum(distances) / len(distances))


def process_frame(source_frame, criteria):
    height, width = source_frame.shape[:2]
    scale = min(1.0, PROCESS_WIDTH / width)
    if scale < 1:
        frame = cv2.resize(
            source_frame, (max(1, int(width * scale)), max(1, int(height * scale)))
        )
    else:
        frame = source_frame.copy()

    people = detect_people(frame)
    reference = criteria["reference_features"]
    has_criteria = bool(criteria["constraints"] or reference)
    matches = []
    for person in people:
        person["appearance_distance"] = None
        person["matches"] = has_criteria and matches_description(
            person, criteria["constraints"]
        )
        if reference:
            person["appearance_distance"] = appearance_distance(
                reference, appearance_descriptor(frame, person["bbox"])
            )
            person["matches"] = (
                person["matches"] and person["appearance_distance"] <= 0.48
            )
        if person["matches"]:
            matches.append(person)

    target = None
    if has_criteria and matches:
        if reference:
            target = min(matches, key=lambda person: person["appearance_distance"])
        else:
            target = max(
                matches,
                key=lambda person: person["bbox"][2] * person["bbox"][3],
            )

    annotated = frame.copy()
    for person in people:
        x, y, box_width, box_height = person["bbox"]
        cv2.rectangle(
            annotated,
            (x, y),
            (x + box_width, y + box_height),
            (0, 0, 255) if person is target else (40, 255, 90),
            3,
        )

    if target is not None:
        x, y, box_width, box_height = target["bbox"]
        pad_x = max(12, int(box_width * 0.3))
        pad_y = max(12, int(box_height * 0.12))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(annotated.shape[1], x + box_width + pad_x)
        bottom = min(annotated.shape[0], y + box_height + pad_y)
        output = annotated[top:bottom, left:right]
    else:
        output = annotated

    success, encoded = cv2.imencode(
        ".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 68]
    )
    if not success:
        raise ValueError("تعذر تجهيز صورة البث.")

    return encoded.tobytes(), people, matches


def process_latest_frames():
    while True:
        FRAME_READY.wait()
        FRAME_READY.clear()
        with STATE_LOCK:
            source_frame = STATE["raw_frame"]
            criteria_version = STATE["criteria_version"]
            criteria = dict(STATE["criteria"])
        if source_frame is None:
            continue
        try:
            frame, people, matches = process_frame(source_frame, criteria)
        except (ValueError, cv2.error):
            app.logger.exception("Could not process camera frame")
            continue
        with STATE_LOCK:
            if criteria_version == STATE["criteria_version"]:
                STATE["frame"] = frame
                STATE["frame_id"] += 1
                STATE["people"] = people
                STATE["matches"] = matches


def ensure_processor():
    global PROCESSOR_THREAD
    with PROCESSOR_LOCK:
        if PROCESSOR_THREAD is None or not PROCESSOR_THREAD.is_alive():
            PROCESSOR_THREAD = threading.Thread(
                target=process_latest_frames, name="latest-frame-processor", daemon=True
            )
            PROCESSOR_THREAD.start()


@app.before_request
def start_background_processor():
    ensure_processor()


@app.get("/")
def index():
    return send_file(WEB_PAGE)


@app.post("/upload")
def upload_frame():
    payload = request.get_json(silent=True)
    frame_data = payload.get("frame") if isinstance(payload, dict) else None
    try:
        frame = decode_image_data(frame_data)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    with STATE_LOCK:
        STATE["raw_frame"] = frame
        STATE["raw_frame_id"] += 1
        frame_id = STATE["raw_frame_id"]
    FRAME_READY.set()
    return jsonify({"accepted": True, "frame_id": frame_id}), 202


@app.get("/api/status")
def status():
    with STATE_LOCK:
        return jsonify({
            "has_frame": STATE["frame"] is not None,
            "frame_id": STATE["frame_id"],
            "people_count": len(STATE["people"]),
            "match_count": len(STATE["matches"]),
        })


@app.get("/api/latest.jpg")
def latest_frame():
    try:
        last_frame_id = int(request.args.get("since", "-1"))
    except ValueError:
        return jsonify({"error": "Invalid frame id."}), 400

    with STATE_LOCK:
        frame = STATE["frame"]
        frame_id = STATE["frame_id"]
        people_count = len(STATE["people"])
        match_count = len(STATE["matches"])
    headers = {
        "X-Frame-Id": str(frame_id),
        "X-People-Count": str(people_count),
        "X-Match-Count": str(match_count),
        "Cache-Control": "no-store",
    }
    if frame is None or frame_id == last_frame_id:
        return Response(status=204, headers=headers)
    return Response(frame, mimetype="image/jpeg", headers=headers)


@app.post("/api/criteria")
def update_criteria():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "أرسل وصف الشخص أو صورة مرجعية."}), 400

    description = payload.get("description", "")
    if not isinstance(description, str):
        return jsonify({"error": "وصف الشخص غير صالح."}), 400
    description = description.strip()
    constraints = parse_description(description)
    image_data = payload.get("reference_image", "")
    if not isinstance(image_data, str):
        return jsonify({"error": "الصورة المرجعية غير صالحة."}), 400
    if not description and not image_data:
        return jsonify({"error": "اكتب مواصفات الشخص أو اختر صورة له."}), 400
    if description and not constraints and not image_data:
        return jsonify({
            "error": "الوصف غير واضح للنظام. اذكر ألوان الملابس مثل: قميص أحمر وبنطلون أسود، أو أرفق صورة.",
        }), 400

    try:
        features = reference_features(image_data) if image_data else None
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    with STATE_LOCK:
        STATE["criteria"] = {
            "description": description,
            "constraints": constraints,
            "reference_features": features,
        }
        STATE["criteria_version"] += 1
        STATE["frame"] = None
        STATE["people"] = []
        STATE["matches"] = []
        response_people = []
        response_matches = []

    FRAME_READY.set()
    return jsonify({
        "people": response_people,
        "matches": response_matches,
        "description": description,
        "reference_image_used": bool(features),
    })


def start_server():
    ensure_processor()
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
