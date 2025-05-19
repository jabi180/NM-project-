# import statements
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
import numpy as np

# Argument parser for ArUco tag type
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50",
                help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

# Supported ArUCo dictionary
ARUCO_DICT = {name: getattr(cv2.aruco, name) for name in dir(cv2.aruco)
              if name.startswith("DICT_")}

if args["type"] not in ARUCO_DICT:
    sys.exit(f"[ERROR] ArUCo tag of '{args['type']}' is not supported")

print(f"[INFO] Detecting '{args['type']}' tags...")

# Load dictionary and detector
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# Camera calibration parameters (replace with actual values)
cameraMatrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1), dtype=np.float32)
markerLength = 0.05  # Marker size in meters

# Start video stream
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

try:
    while True:
        frame = imutils.resize(vs.read(), width=1000)
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)

            for i, marker_id in enumerate(ids.flatten()):
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03)
                topLeft = tuple(corners[i][0][0].astype(int))
                cv2.putText(frame, str(marker_id), (topLeft[0], topLeft[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("AR Marker Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
    vs.stop()
