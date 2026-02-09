import cv2
import numpy as np
import time
import math

# =========================
# CONFIG
# =========================

CAMERA_INDEX = 1

XMAX = 663.0  # mm
YMAX = 316.0  # mm

STABILITY_FRAMES = 12
ANGLE_SMOOTH = 0.8
COLOR_VOTE_FRAMES = 10

# Load homography (PRE-CALIBRATED)
H = np.load("homography.npy")

# =========================
# COLOR DEFINITIONS (HSV)
# =========================

COLOR_RANGES = {
    "RED":   [((0,120,70),(10,255,255)), ((170,120,70),(180,255,255))],
    "GREEN": [((40,70,70),(80,255,255))],
    "BLUE":  [((100,70,70),(130,255,255))]
}

# =========================
# STATE
# =========================

objects = {}
next_id = 0
sent_ids = set()

# =========================
# HELPERS
# =========================

def pixel_to_world(pt):
    px = np.array([[pt]], dtype=np.float32)
    world = cv2.perspectiveTransform(px, H)[0][0]
    return float(world[0]), float(world[1])

def normalize_angle(cnt):
    rect = cv2.minAreaRect(cnt)
    (_, _), (w, h), a = rect
    if w < h:
        a += 90
    return a % 180

def detect_color(hsv, mask):
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        return "UNKNOWN"

    votes = {}
    for color, ranges in COLOR_RANGES.items():
        votes[color] = 0
        for lo, hi in ranges:
            m = cv2.inRange(pixels, np.array(lo), np.array(hi))
            votes[color] += np.count_nonzero(m)

    return max(votes, key=votes.get)

def make_packet(oid, obj):
    x, y = obj["pos"]
    return f"PICK {oid} {x:.1f} {y:.1f} {int(obj['angle'])} {obj['color']}\n"

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Camera not found")

# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        if cv2.contourArea(cnt) < 800:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        wx, wy = pixel_to_world((cx, cy))

        if not (0 <= wx <= XMAX and 0 <= wy <= YMAX):
            continue

        angle = normalize_angle(cnt)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        color = detect_color(hsv, mask)

        matched = None
        for oid, obj in objects.items():
            ox, oy = obj["pos"]
            if abs(wx - ox) < 15 and abs(wy - oy) < 15:
                matched = oid
                break

        if matched is None:
            objects[next_id] = {
                "pos": (wx, wy),
                "angle": angle,
                "angle_frames": 1,
                "color_votes": [color],
                "stable": False
            }
            next_id += 1
            continue

        obj = objects[matched]

        obj["pos"] = (wx, wy)
        obj["angle"] = ANGLE_SMOOTH * obj["angle"] + (1 - ANGLE_SMOOTH) * angle
        obj["angle_frames"] += 1
        obj["color_votes"].append(color)

        if len(obj["color_votes"]) > COLOR_VOTE_FRAMES:
            obj["color_votes"].pop(0)

        if obj["angle_frames"] >= STABILITY_FRAMES:
            obj["stable"] = True

    # =========================
    # SEND READY OBJECTS
    # =========================

    for oid, obj in list(objects.items()):
        if not obj["stable"]:
            continue

        color = max(set(obj["color_votes"]), key=obj["color_votes"].count)
        if color == "UNKNOWN":
            continue

        if oid in sent_ids:
            continue

        obj["color"] = color
        packet = make_packet(oid, obj)
        print(packet.strip())  # <-- SERIAL OUTPUT POINT

        sent_ids.add(oid)

    time.sleep(0.02)
