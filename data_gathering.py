import os
base_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(base_dir,'data')
frame_folder = os.path.join(data_folder,'frames')
os.makedirs(frame_folder,exist_ok=True)
import cv2
import numpy as np
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from screeninfo import get_monitors
cv2.namedWindow("boom", cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow("boom", 1920,0)
cv2.setWindowProperty("boom",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
def primary_monitor_hw():
    main_monitor =[m for m in get_monitors() if m.is_primary][0]
    mH,mW = main_monitor.height, main_monitor.width
    return mH, mW
def random_marker(mH,mW):
    boom = np.ones((mH, mW),dtype=np.uint8) * 255
    x = np.random.randint(0,mW)
    y = np.random.randint(0,mH)
    boom = cv2.circle(boom, (x,y), 4, (0,0,0), -1)
    return boom, (x,y)
def get_date():
    return str(datetime.now()).replace(' ','_').replace(":","-").replace(".","_")
def save_sample(image, coords, data_folder):
    uniqueId = get_date()
    sample_name = f"{uniqueId}_{coords}".replace(" ","")
    cv2.imwrite(os.path.join(data_folder,sample_name+".jpg"),image)
data_gathering = False
if data_gathering :    
    mH, mW = primary_monitor_hw()
    boom, coords = random_marker(mH,mW)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        canvas = image.copy()
        cv2.putText(boom,str(coords),coords,cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)
        cv2.imshow("boom",boom)
        k = cv2.waitKey(5) & 0xFF
        if  k == ord("q"):
            break
        elif k == ord("s"):
            save_sample(image, coords, frame_folder)
            # boom, coords = random_marker(mH,mW)#
        elif k == ord("n"):
            boom, coords = random_marker(mH,mW)
    cv2.destroyAllWindows()
    cap.release()
else:
    import mediapipe as mp
    from asset import detect_faces_mediapipe, GazeEstimator, gaze2point
    def px2cm(coords, width=1920,   height=1080,
                      width_mm=344, height_mm=194):
        x = (coords[0] / width) * width_mm
        x = - x + width_mm / 2
        y = (coords[1] / height) * height_mm
        return (x,y)
    out_dir = os.path.join(data_folder,"faces")
    os.makedirs(out_dir,exist_ok=True)
    checkpoint_path = os.path.join(data_folder,"eth-xgaze_resnet18.pth")
    camera_params = os.path.join(data_folder,"sample_params.yaml")
    normalized_camera_params  = os.path.join(data_folder,"eth-xgaze.yaml")
    estimator = GazeEstimator(checkpoint_path, camera_params, normalized_camera_params)
    detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,static_image_mode=True)
    images_list = glob(os.path.join(data_folder,frame_folder,"*.png"))
    for im_path in images_list:
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        im_name = im_name.split("(")[0]
        coords = eval(im_path[im_path.find("(")+1:im_path.find(")")])
        im = cv2.imread(im_path)
        faces = detect_faces_mediapipe(detector, im)
        face = faces[0]
        estimator.face_model_3d.estimate_head_pose(face, estimator.camera)
        estimator.face_model_3d.compute_3d_pose(face)
        estimator.face_model_3d.compute_face_eye_centers(face, 'ETH-XGaze')
        estimator.head_pose_normalizer.normalize(im, face)
        # find pitch an yaw
        hR = face.head_pose_rot.as_matrix()
        hRx = hR[:, 0]
        forward = (face.center / face.distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R
        gc = np.array(list(px2cm(coords))+[0])
        gc_normalized = gc - face.center * 1000
        print("before_gc_normalized:",gc_normalized)
        gc_normalized = np.dot(R, gc_normalized)
        print("after_gc_normalized:",gc_normalized @ R)
        gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)
        gaze_theta = np.arcsin((-1) * gc_normalized[1])
        gaze_phi = np.arctan2((-1) * gc_normalized[0], (-1) * gc_normalized[2])
        gaze_norm_2d = np.asarray([gaze_theta, gaze_phi])
        pitch, yaw = gaze_norm_2d
        normalized_gaze_vector = -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])

        print("pitch,yaw:",-np.round(gaze_norm_2d,2).tolist()[1])
        print("coords:",np.round(px2cm(coords),2).tolist()[0])
        print("===========")
        cv2.imwrite(f"{out_dir}/face_{im_name}_{str(tuple(gaze_norm_2d.tolist()))}.jpg",face.normalized_image)
