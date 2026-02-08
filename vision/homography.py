import cv2
import cv2.aruco as aruco
import numpy as np

# =====================
# CONFIG
# =====================
CAMERA_INDEX = 1
WIDTH = 1280
HEIGHT = 720

X_MAX = 663.0   # mm
Y_MAX = 316.0   # mm

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

# =====================
# ARUCO
# =====================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict)

WORLD_POINTS = {
    0: (0, 0),
    1: (X_MAX, 0),
    2: (0, Y_MAX),
    3: (X_MAX, Y_MAX),
}

H = None

mouse_x, mouse_y = 0, 0

def mouse_cb(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

cv2.namedWindow("view")
cv2.setMouseCallback("view", mouse_cb)

print("Move camera until all 4 markers are visible.")

# =====================
# LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    img_pts = []
    world_pts = []

    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            if mid in WORLD_POINTS:
                c = corners[i][0]
                center = c.mean(axis=0)

                img_pts.append(center)
                world_pts.append(WORLD_POINTS[mid])

                cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)
                cv2.putText(frame, f"ID {mid}",
                            tuple(center.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        if len(img_pts) == 4 and H is None:
            H, _ = cv2.findHomography(
                np.array(img_pts, np.float32),
                np.array(world_pts, np.float32)
            )
            print("Homography computed")

    # =====================
    # COORD DISPLAY
    # =====================
    if H is not None:
        p = np.array([mouse_x, mouse_y, 1.0])
        P = H @ p
        P /= P[2]

        X, Y = P[0], P[1]

        cv2.circle(frame, (mouse_x, mouse_y), 4, (0,0,255), -1)
        cv2.putText(frame,
                    f"X={X:.1f}mm  Y={Y:.1f}mm",
                    (mouse_x + 10, mouse_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)

    cv2.imshow("view", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
