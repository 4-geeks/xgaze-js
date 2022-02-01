import os
from datetime import datetime
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

base_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(base_dir, 'data')
frame_folder = os.path.join(data_folder, 'camera_calib_frames')
os.makedirs(frame_folder, exist_ok=True)

def get_date():
    return str(datetime.now()).replace(' ', '_').replace(":", "-").replace(".", "_")

def save_sample(image, data_folder, coords="chessboard"):
    uniqueId = get_date()
    sample_name = f"{uniqueId}_{coords}".replace(" ", "")
    cv2.imwrite(os.path.join(data_folder, sample_name+".jpg"), image)
chessboard_size = (9,6)
count = 0
total = 16
data_gathering = False
if __name__ == '__main__':
    if data_gathering:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canvas = image.copy()   
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                # Draw and display the corners
                cv2.drawChessboardCorners(canvas, chessboard_size, corners2, ret)
            cv2.imshow("canvas",canvas)
            k = cv2.waitKey(5) & 0xFF
            if k == ord("q") or count >=total:
                cap.release()
                break

            elif k == ord("s"):
                count += 1
                save_sample(image, frame_folder)
                print(f"[{count}/{total}] frame saved")
    else:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob(os.path.join(frame_folder,'*.jpg'))
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    cv2.destroyAllWindows()
    
image_height,image_width = gray.shape[:2]
camera_matrix = {"rows":mtx.shape[0],
                    "cols":mtx.shape[1],
"data":mtx.tolist()}
distortion_coefficients = {"rows":dist.shape[0],
                    "cols":mtx.shape[1],"data":dist.tolist()}
yml = {"image_height":image_height,"image_width":image_width,
"camera_matrix":camera_matrix,
"distortion_coefficients":distortion_coefficients}

import yaml

with open(r'test.yaml', 'w') as file:
    documents = yaml.dump(yml, file)