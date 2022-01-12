import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from visualizer import Visualizer
from asset import detect_faces_mediapipe, GazeEstimator, gaze2point

checkpoint_path = "../data/eth-xgaze_resnet18.pth"
camera_params = "../data/sample_params.yaml"
normalized_camera_params  = "../data/eth-xgaze.yaml"

estimator = GazeEstimator(checkpoint_path, camera_params, normalized_camera_params)
visualizer = Visualizer(estimator.camera, estimator.face_model_3d.NOSE_INDEX)
detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)


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
    x,y = gaze2point(face.center, face.gaze_vector)
    frame = cv2.flip(frame,1)
    cv2.putText(frame,f"{face.normalized_gaze_angles[0]:0.3f},    {face.normalized_gaze_angles[1]:0.3f}",
        (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    cv2.putText(frame,f"{x},    {y}",
        (50,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    cv2.imshow("frame",frame)
    cv2.imshow("face",cv2.flip(face.normalized_image,1))
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
