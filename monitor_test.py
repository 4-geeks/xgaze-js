import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
from screeninfo import get_monitors
from sympy import Line3D, Matrix, Plane, Point3D

from asset import GazeEstimator, detect_faces_mediapipe, gaze2point
from find_poses import intersection, mm2px

base_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(base_dir, 'data')

with open("poses.txt", "r") as f:
    data = f.read()
x = eval(data)

r = R.from_rotvec(x[3:]).as_matrix()
t = np.array(x[:3])

monitors = get_monitors()
monitor = monitors[0]
mW = monitor.width
width_mm = monitor.width_mm
mH = monitor.height
height_mm = monitor.height_mm

# find 3D points of plane
corners = np.vstack([[0, 0], [mW, 0], [0, mH], [mW, mH]])
corners_3d = (corners / np.array([mW, mH])) * \
    np.array([width_mm, height_mm])
corners_3d = np.hstack([corners_3d, np.zeros((len(corners_3d), 1))])

# convert from mm to meter (m)
corners_3d = corners_3d / 1e3
new_corners = corners_3d @ r + t

# define plane
p1, p2, p3 = new_corners[0], new_corners[1], new_corners[-1]
plane = Plane(Point3D(p1), Point3D(p2), Point3D(p3))
plane_point = list(map(float, (plane.p1.coordinates)))
normal_vector = np.array(list(map(float, plane.normal_vector)))

checkpoint_path = os.path.join(
    data_folder, "finetuned_eth-xgaze_resnet18.pth")
camera_params = os.path.join(data_folder, "sample_params.yaml")
normalized_camera_params = os.path.join(data_folder, "eth-xgaze.yaml")
estimator = GazeEstimator(
    checkpoint_path, camera_params, normalized_camera_params)
detector = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, static_image_mode=True)

cv2.namedWindow("boom", cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow("boom", 1920,0)
cv2.setWindowProperty(
    "boom", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    boom = np.ones((mH, mW), dtype=np.uint8) * 255
    success, image = cap.read()
    if not success:
        continue
    canvas = image.copy()

    faces = detect_faces_mediapipe(detector, image)
    if not len(faces):
        continue

    face = faces[0]
    estimator.estimate(face, image)
    face_center = face.center
    gaze_vector = face.gaze_vector

    intersect = intersection(
        face_center, gaze_vector, plane_point, normal_vector)
    intersect = (intersect - t).reshape(1, 3) @ r
    pog = mm2px(intersect[0][:2]*1e3)
    x, y = list(map(int,pog))
    cv2.circle(boom, (x, y), 7, (86, 55, 15), -1)
    cv2.imshow("boom", boom)
    k = cv2.waitKey(5) & 0xFF
    if k == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()