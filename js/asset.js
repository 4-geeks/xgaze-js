
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
        this.out_mat = new cv.Mat()
        this.dsize = new cv.Size(this.normalized_camera.width, this.normalized_camera.height)
        this.image_mat = new cv.Mat(this.camera.height, this.camera.width, cv.CV_8UC4);
        this.projection_matrix = new cv.Mat(3, 3, cv.CV_64F);

        this.euler_angles2d = new cv.Mat(1, 3, cv.CV_64F);
        this.normalized_head_rot = new cv.Mat(3, 3, cv.CV_64F);
    }

    normalize(image, eye_or_face){
        eye_or_face.normalizing_rot = this._compute_normalizing_rotation(
            eye_or_face.center, eye_or_face.head_pose_rot)
        eye_or_face.normalized_image = this._normalize_image(image, eye_or_face)
        eye_or_face.normalized_head_rot2d = this._normalize_head_pose(eye_or_face)
        return eye_or_face
    }

    _normalize_image(image, eye_or_face){ 
        this.image_mat.data.set(image.data)

        let camera_matrix = math.reshape(math.matrix(this.camera.camera_matrix.data),[3,3])
        let camera_matrix_inv = math.inv(camera_matrix)
        let normalized_camera_matrix = math.reshape(math.matrix(this.normalized_camera.camera_matrix.data),[3,3])
        let scale = this._get_scale_matrix(eye_or_face.distance)
        let conversion_matrix = math.multiply(scale,eye_or_face.normalizing_rot)
        let projection_matrix = math.multiply(math.multiply(normalized_camera_matrix, conversion_matrix), camera_matrix_inv)

        this.projection_matrix.data64F.set(projection_matrix._data.flat())

        cv.warpPerspective(this.image_mat, this.out_mat, this.projection_matrix, this.dsize)
        return this.out_mat
    }

    _normalize_head_pose(eye_or_face){
        let head_pose_rot = math.reshape(math.matrix(Array.from(eye_or_face.head_pose_rot.data64F)),[3,3])
        let normalized_head_rot = math.dotMultiply(head_pose_rot, eye_or_face.normalizing_rot)
        this.normalized_head_rot.data64F.set(normalized_head_rot._data.flat())
        cv.Rodrigues(this.normalized_head_rot, this.euler_angles2d)
        let euler_angles2d  = Array.from(this.euler_angles2d.data64F).slice(0,2)
        let normalized_head_rot2d = math.dotMultiply(euler_angles2d, [1, -1])
        return normalized_head_rot2d
    }

    _compute_normalizing_rotation(center, head_rot){
        let z_axis = _normalize_vector(center)
        let head_x_axis = Array.from([head_rot.data64F[0],head_rot.data64F[3],head_rot.data64F[6]])
        let y_axis = _normalize_vector(math.cross(z_axis, head_x_axis))
        let x_axis = _normalize_vector(math.cross(y_axis, z_axis))
        return [x_axis, y_axis, z_axis]
        
    }

    _get_scale_matrix(distance){

        return [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, math.divide(this.normalized_distance, distance)],
        ]
        
    }
}

async function InferenceONNX(model, data, input_key, w, h){
    const dataFromImage = ndarray(new Float32Array(data), [w, h, 4]);
    const dataProcessed = ndarray(new Float32Array(w * h * 3), [1, 3, w, h]);
    ndarray.ops.divseq(dataFromImage, 255.0);
    ndarray.ops.subseq(dataFromImage.pick(null, null, 0), 0.485);
    ndarray.ops.subseq(dataFromImage.pick(null, null, 1), 0.456);
    ndarray.ops.subseq(dataFromImage.pick(null, null, 2), 0.406);
    ndarray.ops.divseq(dataFromImage.pick(null, null, 0), 0.229);
    ndarray.ops.divseq(dataFromImage.pick(null, null, 1), 0.224);
    ndarray.ops.divseq(dataFromImage.pick(null, null, 2), 0.225);
    ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
    ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 2));
    let input = dataProcessed.data
    const tensor = new ort.Tensor('float32', input, [1, 3, w, h]);
    const output = await model.run({[input_key]:tensor});
    const res = output[Object.keys(output)[0]].data;
    return res;
  }
