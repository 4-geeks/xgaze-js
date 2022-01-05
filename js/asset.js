
function detect_faces_mediapipe(landmarks, imW, imH){
    let pts = []
    for(let i=0;i<landmarks.length;i++){
        pts.push([landmarks[i].x * imW, landmarks[i].y * imH])
    }
    bbox = math.matrix([math.min(pts,0), math.max(pts,0)])
    return new Face(bbox, pts)
}

function _normalize_vector(vector){
    return math.divide(vector, math.norm(vector))
}

class HeadPoseNormalizer {
    constructor(camera, normalized_camera, normalized_distance){
        this.camera = camera
        this.normalized_camera = normalized_camera
        this.normalized_distance
    }

    normalize(image, eye_or_face){
        eye_or_face.normalizing_rot = this._compute_normalizing_rotation(
            eye_or_face.center, eye_or_face.head_pose_rot)
        this._normalize_image(image, eye_or_face)
        this._normalize_head_pose(eye_or_face)
    }

    _normalize_image(image, eye_or_face){
        camera_matrix_inv = math.inv(this.camera.camera_matrix)
        normalized_camera_matrix = this.normalized_camera.camera_matrix

        scale = this._get_scale_matrix(eye_or_face.distance)
        conversion_matrix = math.multiply(scale,eye_or_face.normalizing_rot) // eye_or_face.normalizing_rot.as_matrix()

        projection_matrix = math.multiply(math.multiply(normalized_camera_matrix, conversion_matrix), camera_matrix_inv)

        normalized_image = cv.warpPerspective(
            image, projection_matrix,
            (this.normalized_camera.width, this.normalized_camera.height))

        // if eye_or_face.name in {FacePartsName.REYE, FacePartsName.LEYE}:
        //     normalized_image = cv2.cvtColor(normalized_image,
        //                                     cv2.COLOR_BGR2GRAY)
        //     normalized_image = cv2.equalizeHist(normalized_image)
        eye_or_face.normalized_image = normalized_image

    }

    _normalize_head_pose(eye_or_face){
        normalized_head_rot = math.dotMultiply(eye_or_face.head_pose_rot, eye_or_face.normalizing_rot)
        euler_angles2d = normalized_head_rot.as_euler('XYZ')//[:2]
        eye_or_face.normalized_head_rot2d = math.dotMultiply(euler_angles2d, [1, -1])
    
    }

    _compute_normalizing_rotation(center, head_rot){
        z_axis = _normalize_vector(center.ravel())
        head_rot = head_rot.as_matrix()
        head_x_axis = head_rot//[:, 0]
        y_axis = _normalize_vector(math.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(math.cross(y_axis, z_axis))
        return [x_axis, y_axis, z_axis] // Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))
        
    }

    _get_scale_matrix(distance){
        return [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, math.devide(this.normalized_distance, distance)],
        ]
        
    }


}