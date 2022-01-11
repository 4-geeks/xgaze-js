import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from visualizer import Visualizer
from asset import detect_faces_mediapipe, GazeEstimator

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

    # frame = cv2.flip(frame,1)
    cv2.imshow("frame",frame)
    cv2.imshow("face",face.normalized_image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
