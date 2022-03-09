import csv
import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors

from asset import GazeEstimator, detect_faces_mediapipe, gaze2point, mm2px

base_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(base_dir, 'data')
frame_folder = os.path.join(data_folder, 'monitor_calib_frames')
os.makedirs(frame_folder, exist_ok=True)


def primary_monitor_hw():
    main_monitor = [m for m in get_monitors() if m.is_primary][0]
    mH, mW = main_monitor.height, main_monitor.width
    return mH, mW


def draw_marker(mH, mW, x=None, y=None):
    boom = np.ones((mH, mW), dtype=np.uint8) * 255
    if x is None:
        x = np.random.randint(0, mW)
    if y is None:
        y = np.random.randint(0, mH)
    boom = cv2.circle(boom, (x, y), 4, (0, 0, 0), -1)
    return boom, (x, y)


def corner_pixels(mH, mW, xpad, ypad, row=3, col=3):
    pts = []
    for i in range(row*col):
        r = i % col  # column
        q = i // row  # row
        y = ypad if q == 0 else mH//(row-1) if q == 1 else mH - ypad
        x = xpad if r == 0 else mW//(col-1) if r == 1 else mW - xpad
        pts.append([x, y])
    return np.array(sorted(pts))


def get_date():
    return str(datetime.now()).replace(' ', '_').replace(":", "-").replace(".", "_")


def save_sample(image, coords, data_folder):
    uniqueId = get_date()
    sample_name = f"{uniqueId}_{coords}".replace(" ", "")
    cv2.imwrite(os.path.join(data_folder, sample_name+".jpg"), image)


dataset = []
monitors = get_monitors()
if __name__ == "__main__":
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

    for monitor in monitors:
        mW = monitor.width
        width_mm = monitor.width_mm
        mH = monitor.height
        height_mm = monitor.height_mm
        corners = corner_pixels(mH, mW, xpad=50, ypad=50)
        corner_index = 0

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            canvas = image.copy()

            faces = detect_faces_mediapipe(detector, image)
            if not len(faces):
                continue

            face = faces[0]
            estimator.estimate(face, image)
            x, y = gaze2point(face.center * 1e3, face.gaze_vector)
            x, y = mm2px((x, y))
            boom, coords = draw_marker(
                mH, mW, x=corners[corner_index][0], y=corners[corner_index][1])

            sample = {"coords": list(coords), "face_center": face.center.tolist(),
                      "gaze_vector": face.gaze_vector.tolist()}

            cv2.circle(boom, (x, y), 7, (86, 55, 15), -1)
            cv2.putText(boom, str(coords), coords,
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.imshow("boom", boom)
            k = cv2.waitKey(5) & 0xFF
            if k == ord("q"):
                break

            elif k == ord("s"):  # press 's' to save current frame and coords
                save_sample(image, coords, frame_folder)
                dataset.append(sample)

            elif k == ord("n"):  # press 'n' for going to next point
                corner_index += 1
                if corner_index >= len(corners):
                    break
                boom, coords = draw_marker(
                    mH, mW, x=corners[corner_index][0], y=corners[corner_index][1])

        keys = sample.keys()
        with open('dataset.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(dataset)
        cv2.destroyAllWindows()
        cap.release()
