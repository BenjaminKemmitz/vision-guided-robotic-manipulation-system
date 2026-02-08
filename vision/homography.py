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

XMAX = 663.0   # mm
YMAX = 316.0  # mm

REQUIRED_IDS = [0, 1, 2, 3]

tracked_objects = {}
next_object_id = 0

MAX_MATCH_DIST_MM = 80.0
COLOR_SMOOTHING = 0.8

MAX_MISSED_FRAMES = 5

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
    [0,     0],      # ID 0 bottom-left
    [XMAX,  0],      # ID 1 bottom-right
    [XMAX,  YMAX],   # ID 2 top-right
    [0,     YMAX],   # ID 3 top-left
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

def draw_mm_grid(frame, H, xmax, ymax, minor=10, major=50):
    Hinv = np.linalg.inv(H)

    def w2p(X, Y):
        p = np.array([X, Y, 1.0])
        q = Hinv @ p
        q /= q[2]
        return int(q[0]), int(q[1])

    for x in range(0, int(xmax)+1, minor):
        color = (0,255,0) if x % major == 0 else (80,80,80)
        thick = 2 if x % major == 0 else 1
        cv2.line(frame, w2p(x,0), w2p(x,ymax), color, thick)

    for y in range(0, int(ymax)+1, minor):
        color = (0,255,0) if y % major == 0 else (80,80,80)
        thick = 2 if y % major == 0 else 1
        cv2.line(frame, w2p(0,y), w2p(xmax,y), color, thick)
        
def detect_objects(frame, min_area=1500):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Colorful OR darker-than-table objects
    sat_mask = s > 25        # lowered for green
    val_mask = v < 245       # reject pure white

    mask = np.logical_or(sat_mask, val_mask).astype(np.uint8) * 255

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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

    return objects, mask

def mask_aruco(frame, corners):
    masked = frame.copy()
    if corners is not None:
        for c in corners:
            cv2.fillConvexPoly(masked, c[0].astype(int), (255,255,255))
    return masked

def contour_mean_bgr(frame, cnt):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    b, g, r, _ = cv2.mean(frame, mask)
    return np.array([b, g, r], dtype=np.float32)

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
        IMAGE_POINTS = np.array([
            image_pts[0],
            image_pts[1],
            image_pts[2],
            image_pts[3],
        ], dtype=np.float32)
        H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)
        print("Homography locked")

    if H is not None:
        draw_mm_grid(frame, H, XMAX, YMAX)

        masked = mask_aruco(frame, corners)
        objects, _ = detect_objects(masked)

        updated_ids = set()
        claimed_ids = set()
        updated_ids = set()

        for cx, cy, cnt in objects:
            wx, wy = pixel_to_world(cx, cy, H)
        
            if not (0 <= wx <= XMAX and 0 <= wy <= YMAX):
                continue
        
            mean_bgr = contour_mean_bgr(frame, cnt)
        
            oid = match_object(wx, wy, tracked_objects, claimed_ids)
            claimed_ids.add(oid)
        
            if oid is None:
                oid = next_object_id
                next_object_id += 1
                tracked_objects[oid] = {
                    "pos": (wx, wy),
                    "color": mean_bgr,
                    "missed": 0
                }
            else:
                prev_obj = tracked_objects[oid]
                smooth_color = (
                    COLOR_SMOOTHING * prev_obj["color"]
                    + (1 - COLOR_SMOOTHING) * mean_bgr
                )
                tracked_objects[oid]["pos"] = (wx, wy)
                tracked_objects[oid]["color"] = smooth_color
                tracked_objects[oid]["missed"] = 0
        
            updated_ids.add(oid)
        
            bgr = tracked_objects[oid]["color"]
            b = int(bgr[0])
            g = int(bgr[1])
            r = int(bgr[2])
            hexcol = f"#{r:02X}{g:02X}{b:02X}"
        
            cv2.drawContours(frame, [cnt], -1, (255,0,0), 2)
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
            cv2.putText(
                frame,
                f"ID {oid} {hexcol}",
                (cx+8, cy-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (b,g,r),
                2
            )
        
        # Increment missed counters & prune
        for oid in list(tracked_objects.keys()):
            if oid not in updated_ids:
                tracked_objects[oid]["missed"] += 1
                if tracked_objects[oid]["missed"] > MAX_MISSED_FRAMES:
                    del tracked_objects[oid]


        
        if mouse_x is not None:
            wx, wy = pixel_to_world(mouse_x, mouse_y, H)
            cv2.circle(frame, (mouse_x,mouse_y), 5, (0,0,255), -1)
            cv2.putText(
                frame,
                f"X={wx:.1f} Y={wy:.1f}",
                (mouse_x+10, mouse_y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2
            )

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
