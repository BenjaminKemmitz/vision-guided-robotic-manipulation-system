import cv2
import cv2.aruco as aruco
import numpy as np
import time

# =========================
# CONFIG
# =========================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

XMAX = 663.0
YMAX = 316.0

REQUIRED_IDS = [0, 1, 2, 3]

tracked_objects = {}
next_object_id = 0

MAX_MATCH_DIST_MM = 80.0
COLOR_SMOOTHING = 0.8
MAX_MISSED_FRAMES = 5

STABLE_FRAMES = 8
STABLE_THRESH_MM = 5.0

STATE_NEW = 0
STATE_STABLE = 1
STATE_ASSIGNED = 2
STATE_PICKED = 3

ANGLE_SMOOTHING = 0.7

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 30)

# =========================
# ARUCO
# =========================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, params)

WORLD_POINTS = np.array([
    [0,     0],
    [XMAX,  0],
    [XMAX,  YMAX],
    [0,     YMAX],
], dtype=np.float32)

H = None

# =========================
# MOUSE
# =========================
mouse_x, mouse_y = None, None
def mouse_cb(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_cb)

# =========================
# HELPERS
# =========================
def match_object(wx, wy, tracked, claimed):
    best_id = None
    best_dist = MAX_MATCH_DIST_MM
    for oid, obj in tracked.items():
        if oid in claimed:
            continue
        ox, oy = obj["pos"]
        d = np.hypot(wx - ox, wy - oy)
        if d < best_dist:
            best_dist = d
            best_id = oid
    return best_id

def pixel_to_world(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return world[0][0]

def is_position_stable(history):
    if len(history) < STABLE_FRAMES:
        return False
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return (max(xs) - min(xs) < STABLE_THRESH_MM and
            max(ys) - min(ys) < STABLE_THRESH_MM)

def detect_objects(frame, min_area=1500):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = np.logical_and(s > 20, v < 250).astype(np.uint8) * 255

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        objects.append((cx, cy, c))
    return objects

def contour_mean_bgr(frame, cnt):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    b, g, r, _ = cv2.mean(frame, mask)
    return np.array([b, g, r], dtype=np.float32)

def classify_color_simple(bgr):
    b, g, r = bgr
    if r > g and r > b:
        return "RED"
    if g > r and g > b:
        return "GREEN"
    if b > r and b > g:
        return "BLUE"
    return "UNKNOWN"

def update_color_vote(obj, bgr, min_votes=8):
    label = classify_color_simple(bgr)
    if label == "UNKNOWN":
        return

    obj["color_history"].append(label)

    # keep history bounded
    if len(obj["color_history"]) > min_votes:
        obj["color_history"].pop(0)

    # only lock color once object is STABLE
    if obj["state"] != STATE_STABLE:
        return

    if len(obj["color_history"]) < min_votes:
        return

    # majority vote
    counts = {}
    for c in obj["color_history"]:
        counts[c] = counts.get(c, 0) + 1

    winner = max(counts, key=counts.get)

    # require strong majority
    if counts[winner] >= int(0.7 * min_votes):
        obj["color_label"] = winner

def circular_mean(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    sin_sum = np.mean(np.sin(angles_rad))
    cos_sum = np.mean(np.cos(angles_rad))
    mean = np.arctan2(sin_sum, cos_sum)
    return np.rad2deg(mean) % 180

# =========================
# FPS
# =========================
prev = time.perf_counter()
fc = 0
fps = 0.0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    image_pts = {}
    if ids is not None:
        ids = ids.flatten()
        aruco.drawDetectedMarkers(frame, corners, ids)
        for c, mid in zip(corners, ids):
            image_pts[mid] = c[0].mean(axis=0)

    if H is None and all(i in image_pts for i in REQUIRED_IDS):
        IMAGE_POINTS = np.array([image_pts[i] for i in REQUIRED_IDS], dtype=np.float32)
        H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)
        print("Homography locked")

    if H is not None:
        objects = detect_objects(frame)

        claimed_ids = set()
        updated_ids = set()

        for cx, cy, cnt in objects:
            wx, wy = pixel_to_world(cx, cy, H)
            if not (0 <= wx <= XMAX and 0 <= wy <= YMAX):
                continue

            mean_bgr = contour_mean_bgr(frame, cnt)
            oid = match_object(wx, wy, tracked_objects, claimed_ids)

            rect = cv2.minAreaRect(cnt)
            (_, _), (w, h), raw_angle = rect

            # OpenCV angle correction
            if w < h:
                angle = raw_angle + 90
            else:
                angle = raw_angle
            
            # Force angle into [0, 180)
            angle = angle % 180       


            if oid is None:
                oid = next_object_id
                next_object_id += 1
                tracked_objects[oid] = {
                    "pos": (wx, wy),
                    "color": mean_bgr,
                    "missed": 0,
                    "state": STATE_NEW,
                    "age": 1,
                    "pos_history": [(wx, wy)],
                    "angle": angle,
                    "color_history": [],
                    "color_label": "UNKNOWN",
                    "angle_history": [],
                    "angle_locked": False
                }
            else:
                obj = tracked_objects[oid]
                obj["pos"] = (wx, wy)
                obj["color"] = COLOR_SMOOTHING * obj["color"] + (1 - COLOR_SMOOTHING) * mean_bgr
                obj["angle_history"].append(angle)
                if len(obj["angle_history"]) > STABLE_FRAMES:
                    obj["angle_history"].pop(0)
                
                # Lock angle once stable
                if obj["state"] == STATE_STABLE and not obj["angle_locked"]:
                    if len(obj["angle_history"]) == STABLE_FRAMES:
                        obj["angle"] = circular_mean(obj["angle_history"])
                        obj["angle_locked"] = True
                
                obj["missed"] = 0
                obj["age"] += 1
                obj["pos_history"].append((wx, wy))
                if len(obj["pos_history"]) > STABLE_FRAMES:
                    obj["pos_history"].pop(0)
                if obj["state"] == STATE_NEW and is_position_stable(obj["pos_history"]):
                    obj["state"] = STATE_STABLE
                update_color_vote(obj, obj["color"])

            claimed_ids.add(oid)
            updated_ids.add(oid)

            obj = tracked_objects[oid]
            state_color = (0,255,0) if obj["state"] == STATE_STABLE else (0,255,255)

            cv2.drawContours(frame, [cnt], -1, state_color, 2)
            theta = np.deg2rad(obj["angle"])
            length = 30
            x2 = int(cx + length * np.cos(theta))
            y2 = int(cy + length * np.sin(theta))
            cv2.line(frame, (cx, cy), (x2, y2), (255, 255, 255), 2)
            
            if obj["angle_locked"]:
                angle_text = f"{int(obj['angle'])}°"
            else:
                angle_text = "??"
            
            label = f"ID {oid} {obj['color_label']} {angle_text}"

            
            cv2.putText(
                frame,
                label,
                (cx+8, cy-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0) if obj["color_label"] != "UNKNOWN" else (0,255,255),
                2
            )


        for oid in list(tracked_objects.keys()):
            if oid not in updated_ids:
                tracked_objects[oid]["missed"] += 1
                if tracked_objects[oid]["missed"] > MAX_MISSED_FRAMES:
                    del tracked_objects[oid]

    fc += 1
    now = time.perf_counter()
    if now-prev >= 1.0:
        fps = fc/(now-prev)
        fc = 0
        prev = now

    cv2.putText(frame, f"FPS {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
