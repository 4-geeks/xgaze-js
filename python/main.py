import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from visualizer import Visualizer
from asset import detect_faces_mediapipe, GazeEstimator, gaze2point, mm2px
cv2.namedWindow("boom", cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow("boom", 1920,0)
cv2.setWindowProperty("boom",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
checkpoint_path = "../data/finetuned_eth-xgaze_resnet18.pth"
camera_params = "../data/sample_params.yaml"
normalized_camera_params  = "../data/eth-xgaze.yaml"

estimator = GazeEstimator(checkpoint_path, camera_params, normalized_camera_params)
visualizer = Visualizer(estimator.camera, estimator.face_model_3d.NOSE_INDEX)
detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

xy_list = []
boom = np.ones((1080,1920))
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret :
        break

    faces = detect_faces_mediapipe(detector, frame)
    visualizer.set_image(frame)
    for face in faces:
        estimator.estimate(face, frame)
        visualizer.draw_3d_line(face.center, face.center + 0.5 * face.gaze_vector)
    x,y = gaze2point(face.center * 1e3, face.gaze_vector)
    x,y = mm2px((x,y))
    xy_list.append([x,y])
    xy_list = xy_list[-10:]
    x,y = np.mean(xy_list,0).astype(np.int)
    canvas = cv2.circle(boom.copy(),(x,y),10,(0,0,0),-1)
    frame = cv2.flip(frame,1)
    cv2.putText(canvas,f"{x},  {y}",
        (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    cv2.imshow("boom",canvas)
    cv2.imshow("face",cv2.flip(face.normalized_image,1))
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
