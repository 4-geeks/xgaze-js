import os
from datetime import datetime
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from screeninfo import get_monitors

base_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(base_dir, 'data')
frame_folder = os.path.join(data_folder, 'monitor_calib_frames')
os.makedirs(frame_folder, exist_ok=True)
cv2.namedWindow("boom", cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow("boom", 1920,0)
cv2.setWindowProperty("boom", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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
    return np.array(pts)


def get_date():
    return str(datetime.now()).replace(' ', '_').replace(":", "-").replace(".", "_")


def save_sample(image, coords, data_folder):
    uniqueId = get_date()
    sample_name = f"{uniqueId}_{coords}".replace(" ", "")
    cv2.imwrite(os.path.join(data_folder, sample_name+".jpg"), image)


data_gathering = False
monitors = get_monitors()
if __name__ == "__main__":
    for monitor in monitors:
        mW = monitor.width
        width_mm = monitor.width_mm
        mH = monitor.height
        height_mm = monitor.height_mm
        corners = corner_pixels(mH, mW, xpad=50, ypad=50)
        corner_index = 0
        boom, coords = draw_marker(
            mH, mW, x=corners[corner_index][0], y=corners[corner_index][0])

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            canvas = image.copy()
            cv2.putText(boom, str(coords), coords,
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.imshow("boom", boom)
            k = cv2.waitKey(5) & 0xFF
            if k == ord("q"):
                break

            elif k == ord("s"): # press 's' to save current frame and coords
                save_sample(image, coords, frame_folder)

            elif k == ord("n"): # press 'n' for going to next point
                corner_index += 1
                boom, coords = draw_marker(
                    mH, mW, x=corners[corner_index][0], y=corners[corner_index][0])

        cv2.destroyAllWindows()
        cap.release()
