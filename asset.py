import os
import cv2
import timm
import torch
import urllib
import numpy as np
import torchvision.transforms as T
from scipy.spatial.transform import Rotation
from face_cam import Face, Camera, FaceParts, FacePartsName, FaceModelMediaPipe, Camera

from model import Model

transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, (224, 224))),
        T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                     0.225]),  # RGB
    ])
# gazetr_transform = T.Compose([
#         T.Lambda(lambda x: cv2.resize(x, (224, 224))),
#         T.ToTensor()
#     ])
# gazetr_model = Model()
# state_dict = torch.load("GazeTR-H-ETH.pt",map_location="cuda")
# gazetr_model.cuda()
# gazetr_model.load_state_dict(state_dict)
# gazetr_model.eval()
def detect_faces_mediapipe(detector, image: np.ndarray):
        h, w = image.shape[:2]
        predictions = detector.process(image[:, :, ::-1])
        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts))
        return detected


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


class HeadPoseNormalizer:
    def __init__(self, camera: Camera, normalized_camera: Camera,
                 normalized_distance: float):
        self.camera = camera
        self.normalized_camera = normalized_camera
        self.normalized_distance = normalized_distance

    def normalize(self, image: np.ndarray, eye_or_face: FaceParts) -> None:
        eye_or_face.normalizing_rot = self._compute_normalizing_rotation(
            eye_or_face.center, eye_or_face.head_pose_rot)
        self._normalize_image(image, eye_or_face)
        self._normalize_head_pose(eye_or_face)

    def _normalize_image(self, image: np.ndarray,
                         eye_or_face: FaceParts) -> None:
        camera_matrix_inv = np.linalg.inv(self.camera.camera_matrix)
        normalized_camera_matrix = self.normalized_camera.camera_matrix

        scale = self._get_scale_matrix(eye_or_face.distance)
        conversion_matrix = scale @ eye_or_face.normalizing_rot.as_matrix()

        projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv

        normalized_image = cv2.warpPerspective(
            image, projection_matrix,
            (self.normalized_camera.width, self.normalized_camera.height))

        if eye_or_face.name in {FacePartsName.REYE, FacePartsName.LEYE}:
            normalized_image = cv2.cvtColor(normalized_image,
                                            cv2.COLOR_BGR2GRAY)
            normalized_image = cv2.equalizeHist(normalized_image)
        eye_or_face.normalized_image = normalized_image

    @staticmethod
    def _normalize_head_pose(eye_or_face: FaceParts) -> None:
        normalized_head_rot = eye_or_face.head_pose_rot * eye_or_face.normalizing_rot
        euler_angles2d = normalized_head_rot.as_euler('XYZ')[:2]
        eye_or_face.normalized_head_rot2d = euler_angles2d * np.array([1, -1])

    @staticmethod
    def _compute_normalizing_rotation(center: np.ndarray,
                                      head_rot: Rotation) -> Rotation:
        z_axis = _normalize_vector(center.ravel())
        head_rot = head_rot.as_matrix()
        head_x_axis = head_rot[:, 0]
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        return Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))

    def _get_scale_matrix(self, distance: float) -> np.ndarray:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, self.normalized_distance / distance],
        ],
                        dtype=np.float)


class GazeEstimator:
    def __init__(self, checkpoint_path:str, camera_params: str, normalized_camera_params: str, model_name :str = "resnet18",
                 normalized_distance: float =0.6, device: str = None):
        camera = Camera(camera_params)
        normalized_camera = Camera(normalized_camera_params)
        self.camera = camera
        self.normalized_camera = normalized_camera
        self.face_model_3d = FaceModelMediaPipe()
        self.head_pose_normalizer = HeadPoseNormalizer(camera, normalized_camera, normalized_distance=normalized_distance)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.gaze_estimation_model = timm.create_model(model_name, num_classes=2)
        try:
            if not os.path.isfile(checkpoint_path):
                g = urllib.request.urlopen('https://github.com/4-geeks/xgaze-js/releases/download/v0.0.1/eth-xgaze_resnet18.pth')
                with open(checkpoint_path, 'b+w') as f:
                    f.write(g.read())
            checkpoint = torch.load(checkpoint_path,map_location='cpu')
            self.gaze_estimation_model.load_state_dict(checkpoint['model'])
            print("model successfully loaded.")
        except:
            print("model loading failed.")
        self.gaze_estimation_model.to(device)
        self.gaze_estimation_model.eval()

    def estimate(self, face, frame):
        self.face_model_3d.estimate_head_pose(face, self.camera)
        self.face_model_3d.compute_3d_pose(face)
        self.face_model_3d.compute_face_eye_centers(face, 'ETH-XGaze')
        self.head_pose_normalizer.normalize(frame, face)
        image = transform(face.normalized_image).unsqueeze(0)
        image = image.to(self.device)
        prediction = self.gaze_estimation_model(image)
        prediction = prediction.detach().cpu().numpy()
        # image = gazetr_transform(face.normalized_image).unsqueeze(0)
        # image = image.to(self.device)
        # prediction = gazetr_model({"face":image})
        # prediction = prediction.detach().cpu().numpy()[...,::-1]
        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()   

def gaze2point(center, gaze_vector):
    """
    x = gaze_vector[0] * t + center[0]
    y = gaze_vector[1] * t + center[1]
    z = gaze_vector[2] * t + center[2]

    solve it for z=0 :
    """
    t = - center[2] / gaze_vector[2]
    x = gaze_vector[0] * t + center[0]
    y = gaze_vector[1] * t + center[1]
    return x,y

def px2mm(coords, width=1920,   height=1080,
                      width_mm=344, height_mm=194):
        x = (coords[0] / width) * width_mm
        x = - x + width_mm / 2
        y = (coords[1] / height) * height_mm
        return (x,y)

def mm2px(point, width=1920,   height=1080,
                 width_mm=344, height_mm=194):
    x,y = point
    x = - x + width_mm / 2
    x = (x / width_mm) * width
    y = (y / height_mm) * height
    return round(x),round(y)
