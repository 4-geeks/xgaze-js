
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
        this.normalized_distance = normalized_distance
    }

    normalize(image, eye_or_face){
        eye_or_face.normalizing_rot = this._compute_normalizing_rotation(
            eye_or_face.center, eye_or_face.head_pose_rot)
        eye_or_face.normalized_image = this._normalize_image(image, eye_or_face)
        eye_or_face.normalized_head_rot2d = this._normalize_head_pose(eye_or_face)
        return eye_or_face
    }

    _normalize_image(image, eye_or_face){
        var image_mat = new cv.Mat(image.height, image.width, cv.CV_8UC4);
        image_mat.data.set(image.data)
        var out_mat = new cv.Mat()
        let dsize = new cv.Size(this.normalized_camera.width, this.normalized_camera.height)
        let camera_matrix = math.reshape(math.matrix(this.camera.camera_matrix.data),[3,3])
        let camera_matrix_inv = math.inv(camera_matrix)
        let normalized_camera_matrix = math.reshape(math.matrix(this.normalized_camera.camera_matrix.data),[3,3])
        
        let scale = this._get_scale_matrix(eye_or_face.distance)
        let conversion_matrix = math.multiply(scale,eye_or_face.normalizing_rot) // eye_or_face.normalizing_rot.as_matrix()

        let projection_matrix = math.multiply(math.multiply(normalized_camera_matrix, conversion_matrix), camera_matrix_inv)
        
        projection_matrix = new cv.matFromArray(projection_matrix._size[0], projection_matrix._size[1],
                                                cv.CV_64FC1, projection_matrix._data.flat())
        
        cv.warpPerspective(image_mat, out_mat, projection_matrix, dsize)
        return out_mat
    }

    _normalize_head_pose(eye_or_face){
        let head_pose_rot = math.reshape(math.matrix(Array.from(eye_or_face.head_pose_rot.data64F)),[3,3])
        let normalized_head_rot = math.dotMultiply(head_pose_rot, eye_or_face.normalizing_rot)
        let normalized_head_rota_mat = new cv.matFromArray(normalized_head_rot._size[0], normalized_head_rot._size[1],
                                                            cv.CV_64FC1, normalized_head_rot._data.flat())
        var euler_angles2d = new cv.Mat({ width: 3, height: 1 }, cv.CV_64FC1);
        cv.Rodrigues(normalized_head_rota_mat, euler_angles2d)
        euler_angles2d  = Array.from(euler_angles2d.data64F).slice(0,2)
        let normalized_head_rot2d = math.dotMultiply(euler_angles2d, [1, -1])
        return normalized_head_rot2d
    }

    _compute_normalizing_rotation(center, head_rot){
        let z_axis = _normalize_vector(center)
        // head_rot = head_rot.as_matrix()
        let head_x_axis = Array.from([head_rot.data64F[0],head_rot.data64F[3],head_rot.data64F[6]])
        let y_axis = _normalize_vector(math.cross(z_axis, head_x_axis))
        let x_axis = _normalize_vector(math.cross(y_axis, z_axis))
        return [x_axis, y_axis, z_axis] // Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))
        
    }

    _get_scale_matrix(distance){

        return [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, math.divide(this.normalized_distance, distance)],
        ]
        
    }


}