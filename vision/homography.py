import cv2
import cv2.aruco as aruco
import numpy as np
import time

# =========================
# CONFIGURATION
# =========================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

XMAX = 663.0   # mm
YMAX = 316.0  # mm

REQUIRED_IDS = [0, 1, 2, 3]

def draw_mm_grid(frame, H, xmax, ymax,
                 minor=10, major=50):
    """
    Draws a metric grid (mm) onto the frame using homography.
    """
    Hinv = np.linalg.inv(H)
    h, w = frame.shape[:2]

    def world_to_pixel(X, Y):
        p = np.array([X, Y, 1.0])
        q = Hinv @ p
        q /= q[2]
        return int(q[0]), int(q[1])

    # ----- Vertical lines (X constant)
    for x in range(0, int(xmax) + 1, minor):
        color = (80, 80, 80) if x % major else (0, 255, 0)
        thickness = 1 if x % major else 2

        try:
            p1 = world_to_pixel(x, 0)
            p2 = world_to_pixel(x, ymax)
            cv2.line(frame, p1, p2, color, thickness)

            if x % major == 0:
                cv2.putText(frame, f"{x}",
                            (p1[0] + 2, p1[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)
        except:
            pass

    # ----- Horizontal lines (Y constant)
    for y in range(0, int(ymax) + 1, minor):
        color = (80, 80, 80) if y % major else (0, 255, 0)
        thickness = 1 if y % major else 2

        try:
            p1 = world_to_pixel(0, y)
            p2 = world_to_pixel(xmax, y)
            cv2.line(frame, p1, p2, color, thickness)

            if y % major == 0:
                cv2.putText(frame, f"{y}",
                            (p1[0] + 2, p1[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)
        except:
            pass

def detect_objects(frame, min_area=500):
    """
    Returns list of (cx, cy, contour)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        objects.append((cx, cy, cnt))

    return objects, thresh

def pixel_to_world(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return world[0][0][0], world[0][0][1]

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 30)

# =========================
# ARUCO SETUP
# =========================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, params)

# =========================
# WORLD POINTS
# =========================
WORLD_POINTS = np.array([
    [0,     0],
    [XMAX,  0],
    [XMAX,  YMAX],
    [0,     YMAX],
], dtype=np.float32)

H = None

# =========================
# FPS TRACKING
# =========================
prev_time = time.perf_counter()
frame_count = 0
fps = 0.0

# =========================
# MOUSE STATE
# =========================
mouse_x, mouse_y = None, None

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", mouse_callback)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    image_points = {}

    if ids is not None:
        ids = ids.flatten()
        aruco.drawDetectedMarkers(frame, corners, ids)

        for corner, marker_id in zip(corners, ids):
            center = corner[0].mean(axis=0)
            image_points[marker_id] = center

            cx, cy = int(center[0]), int(center[1])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {marker_id}",
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

    if H is None and all(i in image_points for i in REQUIRED_IDS):
        IMAGE_POINTS = np.array([
            image_points[0],
            image_points[1],
            image_points[2],
            image_points[3],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)
        print("Homography computed!")

    # Draw mouse world coordinates
    if H is not None and mouse_x is not None:
        pt = np.array([[[mouse_x, mouse_y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, H)
        wx, wy = world[0][0]

        cv2.circle(frame, (mouse_x, mouse_y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"X={wx:.1f} mm  Y={wy:.1f} mm",
            (mouse_x + 10, mouse_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
    if H is not None:
        draw_mm_grid(frame, H, XMAX, YMAX)
    if H is not None:
    objects, thresh = detect_objects(frame)

    for i, (cx, cy, cnt) in enumerate(objects):
        wx, wy = pixel_to_world(cx, cy, H)

        # Draw contour
        cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)

        # Draw centroid
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Label object
        cv2.putText(
            frame,
            f"Obj {i}: X={wx:.1f} Y={wy:.1f}",
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # FPS
    frame_count += 1
    now = time.perf_counter()
    if now - prev_time >= 1.0:
        fps = frame_count / (now - prev_time)
        frame_count = 0
        prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
