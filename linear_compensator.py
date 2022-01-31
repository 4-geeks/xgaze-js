import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from torch import nn

from asset import GazeEstimator, detect_faces_mediapipe, gaze2point, mm2px

checkpoint_path = "./data/finetuned_eth-xgaze_resnet18.pth"
camera_params = "./data/sample_params.yaml"
normalized_camera_params = "./data/eth-xgaze.yaml"

estimator = GazeEstimator(checkpoint_path, camera_params,
                          normalized_camera_params, model_name="resnet18")
detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True)


class RegNet(nn.Module):
    def __init__(self, input_features=2, output_features=2):
        super(RegNet, self).__init__()
        self.layer1 = nn.Linear(input_features, 16)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(16, output_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x

width=1920
height=1080
images_list = sorted(glob("./data/frames/*.jpg") + glob("./data/frames/*.png"))
X, y = [], []
for im_path in images_list:
    frame = cv2.imread(im_path)
    im_name = os.path.splitext(os.path.basename(im_path))[0]
    im_name = im_name.split("(")[0]
    coords = eval(im_path[im_path.find("(")+1:im_path.find(")")])
    faces = detect_faces_mediapipe(detector, frame)
    for face in faces:
        estimator.estimate(face, frame)
    head_pose_rot = face.head_pose_rot.as_rotvec()
    head_position = face.head_position
    u, v = gaze2point(face.center * 1e3, face.gaze_vector)
    u, v = mm2px((u, v),width=width,height=height)
    X.append([u/width, v/height, *head_pose_rot, *head_position])
    y.append([coords[0]/width, coords[1]/height])
X = np.array(X)
y = np.array(y)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from skorch.regressor import NeuralNetRegressor
    import pickle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print("before training acc in pixels:",np.abs(y_test - X_test[:,:2]).mean(0)*np.array([width,height]))
    training = False
    if training:
        with open('data/scaler.pkl','rb') as f:
            scaler = pickle.load(f)
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        myNet = RegNet(input_features=X_train.shape[1],output_features=2)
        net = NeuralNetRegressor(
            myNet,
            max_epochs=100,
            lr=0.01,
            iterator_train__shuffle=True,
        )
        net.fit(X_train, y_train.astype(np.float32))
        net.save_params(f_params='data/model.pkl',
                f_optimizer='data/opt.pkl', f_history='data/history.json')
        y_proba = net.predict_proba(X_test)
        print("before training acc in pixels:",np.abs(y_test - y_proba).mean(0)* np.array([width,height]))
