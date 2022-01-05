
const normalized_distance = 0.6
const camera_params = "./sample_params.yaml"
const normalized_camera_params  = "./eth-xgaze.yaml"
const camera = new Camera(camera_params)
const normalized_camera = new Camera(normalized_camera_params)
const face_model_3d = new FaceModelMediaPipe()
const head_pose_normalizer = new HeadPoseNormalizer(camera, normalized_camera, normalized_distance)
const [imW, imH] = [camera.width, camera.height]
async function onResults(results) {
    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            face = detect_faces_mediapipe(landmarks, imW, imH)
            face = face_model_3d.estimate_head_pose(face, camera)
            face = face_model_3d.compute_3d_pose(face)
            face = face_model_3d.compute_face_eye_centers(face)
            head_pose_normalizer.normalize(frame, face)
        }
    }
}

const faceMesh = new FaceMesh({locateFile: (path) => {
    return `dist/${path}`;
  }});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
faceMesh.onResults(onResults);
async function Run(){
    let imageSize = 480
    const imageLoader = new ImageLoader(imageSize, imageSize);
    console.log("imageLoader",imageLoader)
    const imageData = await imageLoader.getImageData('./test.jpg');
    console.log("imageData",imageData)
    await faceMesh.send({image: imageData})
}
Run()
