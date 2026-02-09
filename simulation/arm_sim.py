import cv2
import cv2.aruco as aruco
import numpy as np
import time

# =====================================================
# CONFIG
# =====================================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

XMAX = 663.0
YMAX = 316.0

REQUIRED_IDS = [0, 1, 2, 3]

MAX_MATCH_DIST_MM = 80.0
MAX_MISSED_FRAMES = 5

STABLE_FRAMES = 8
STABLE_THRESH_MM = 5.0

COLOR_SMOOTHING = 0.8

STATE_NEW = 0
STATE_STABLE = 1
STATE_ASSIGNED = 2
STATE_PICKED = 3

# =====================================================
# FAKE ROBOT CONFIG
# =====================================================
ROBOT_HOME = np.array([XMAX / 2, YMAX + 80])
ROBOT_SPEED = 150.0

DROP_ZONES = {
    "RED":   np.array([50,  -40]),
    "GREEN": np.array([330, -40]),
    "BLUE":  np.array([610, -40]),
}

robot_pos = ROBOT_HOME.copy()
robot_target = None
robot_state = "IDLE"
robot_object_id = None
last_robot_time = time.perf_counter()

# =====================================================
# CAMERA
# =====================================================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

# =====================================================
# ARUCO
# =====================================================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

WORLD_POINTS = np.array([
    [0, 0],
    [XMAX, 0],
    [XMAX, YMAX],
    [0, YMAX]
], dtype=np.float32)

H = None

# =====================================================
# TRACKING
# =====================================================
tracked_objects = {}
next_object_id = 0

# =====================================================
# HELPERS
# =====================================================
def pixel_to_world(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]

def is_position_stable(history):
    if len(history) < STABLE_FRAMES:
        return False
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return max(xs) - min(xs) < STABLE_THRESH_MM and max(ys) - min(ys) < STABLE_THRESH_MM

def circular_mean(angles):
    r = np.deg2rad(angles)
    return np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 180

def classify_color(bgr):
    b, g, r = bgr
    if r > g and r > b: return "RED"
    if g > r and g > b: return "GREEN"
    if b > r and b > g: return "BLUE"
    return "UNKNOWN"

def choose_pick_candidate():
    for oid, obj in tracked_objects.items():
        if obj["state"] == STATE_STABLE and obj["color_label"] != "UNKNOWN" and obj["angle_locked"]:
            return oid
    return None

def move_towards(cur, tgt, speed, dt):
    d = tgt - cur
    dist = np.linalg.norm(d)
    if dist < 1e-3:
        return tgt.copy(), True
    step = min(speed * dt, dist)
    return cur + d / dist * step, step >= dist

# =====================================================
# MAIN LOOP
# =====================================================
prev_fps_time = time.perf_counter()
fps_counter = 0
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # ---------------- ARUCO ----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    image_pts = {}
    if ids is not None:
        ids = ids.flatten()
        aruco.drawDetectedMarkers(frame, corners, ids)
        for c, mid in zip(corners, ids):
            image_pts[mid] = c[0].mean(axis=0)

    if H is None and all(i in image_pts for i in REQUIRED_IDS):
        img_pts = np.array([image_pts[i] for i in REQUIRED_IDS], dtype=np.float32)
        H, _ = cv2.findHomography(img_pts, WORLD_POINTS)
        print("Homography locked")

    # ---------------- DETECTION ----------------
    if H is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s, v = hsv[:,:,1], hsv[:,:,2]
        mask = ((s > 20) & (v < 250)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        updated = set()

        for cnt in contours:
            if cv2.contourArea(cnt) < 1500:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            wx, wy = pixel_to_world(cx, cy, H)
            if not (0 <= wx <= XMAX and 0 <= wy <= YMAX):
                continue

            mean_bgr = np.array(cv2.mean(frame, cv2.drawContours(
                np.zeros(frame.shape[:2], np.uint8), [cnt], -1, 255, -1))[:3])

            rect = cv2.minAreaRect(cnt)
            (_, _), (w, h), raw = rect
            angle = (raw + 90 if w < h else raw) % 180

            # match or create
            oid = None
            for i, o in tracked_objects.items():
                if np.hypot(wx-o["pos"][0], wy-o["pos"][1]) < MAX_MATCH_DIST_MM:
                    oid = i
                    break

            if oid is None:
                tracked_objects[next_object_id] = {
                    "pos": (wx, wy),
                    "pos_history": [(wx, wy)],
                    "color": mean_bgr,
                    "color_history": [],
                    "color_label": "UNKNOWN",
                    "angle_history": [],
                    "angle": angle,
                    "angle_locked": False,
                    "state": STATE_NEW,
                    "missed": 0
                }
                oid = next_object_id
                next_object_id += 1
            else:
                obj = tracked_objects[oid]
                obj["pos"] = (wx, wy)
                obj["pos_history"].append((wx, wy))
                obj["pos_history"] = obj["pos_history"][-STABLE_FRAMES:]

                obj["angle_history"].append(angle)
                obj["angle_history"] = obj["angle_history"][-STABLE_FRAMES:]

                obj["color"] = COLOR_SMOOTHING * obj["color"] + (1-COLOR_SMOOTHING)*mean_bgr
                obj["color_history"].append(classify_color(obj["color"]))
                obj["color_history"] = obj["color_history"][-STABLE_FRAMES:]

                if obj["state"] == STATE_NEW and is_position_stable(obj["pos_history"]):
                    obj["state"] = STATE_STABLE

                if obj["state"] == STATE_STABLE and not obj["angle_locked"]:
                    obj["angle"] = circular_mean(obj["angle_history"])
                    obj["angle_locked"] = True

                if obj["state"] == STATE_STABLE and obj["color_label"] == "UNKNOWN":
                    counts = {c:obj["color_history"].count(c) for c in obj["color_history"]}
                    winner = max(counts, key=counts.get)
                    if counts[winner] >= 6:
                        obj["color_label"] = winner

                obj["missed"] = 0

            updated.add(oid)

            # draw
            obj = tracked_objects[oid]
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
            cv2.putText(frame,
                        f"ID {oid} {obj['color_label']} {int(obj['angle']) if obj['angle_locked'] else '??'}°",
                        (cx+8, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        for oid in list(tracked_objects.keys()):
            if oid not in updated:
                tracked_objects[oid]["missed"] += 1
                if tracked_objects[oid]["missed"] > MAX_MISSED_FRAMES:
                    del tracked_objects[oid]

    # ---------------- ROBOT STATE MACHINE ----------------
    now = time.perf_counter()
    dt = now - last_robot_time
    last_robot_time = now

    if robot_state == "IDLE":
        pick = choose_pick_candidate()
        if pick is not None:
            robot_object_id = pick
            robot_target = np.array(tracked_objects[pick]["pos"])
            tracked_objects[pick]["state"] = STATE_ASSIGNED
            robot_state = "MOVE_TO_PICK"

    elif robot_state == "MOVE_TO_PICK":
        robot_pos, done = move_towards(robot_pos, robot_target, ROBOT_SPEED, dt)
        if done:
            robot_state = "PICK"

    elif robot_state == "PICK":
        tracked_objects[robot_object_id]["state"] = STATE_PICKED
        robot_target = DROP_ZONES[tracked_objects[robot_object_id]["color_label"]]
        robot_state = "MOVE_TO_DROP"

    elif robot_state == "MOVE_TO_DROP":
        robot_pos, done = move_towards(robot_pos, robot_target, ROBOT_SPEED, dt)
        if done:
            del tracked_objects[robot_object_id]
            robot_object_id = None
            robot_state = "IDLE"

    # ---------------- DRAW ROBOT ----------------
    rx, ry = map(int, robot_pos)
    cv2.circle(frame, (rx, ry), 14, (255,255,255), -1)
    cv2.putText(frame, robot_state, (rx-40, ry-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # ---------------- FPS ----------------
    fps_counter += 1
    if now - prev_fps_time >= 1.0:
        fps = fps_counter / (now - prev_fps_time)
        fps_counter = 0
        prev_fps_time = now

    cv2.putText(frame, f"FPS {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
