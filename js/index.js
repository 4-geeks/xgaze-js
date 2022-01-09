var imageData;
const normalized_distance = 0.6
const camera_params = "./sample_params.yaml"
const normalized_camera_params  = "./eth-xgaze.yaml"
const camera = new Camera(camera_params)
const normalized_camera = new Camera(normalized_camera_params)
const face_model_3d = new FaceModelMediaPipe()
const head_pose_normalizer = new HeadPoseNormalizer(camera, normalized_camera, normalized_distance)
const [imW, imH] = [1200,800] //[camera.width, camera.height]
async function onResults(results) {
    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            face = detect_faces_mediapipe(landmarks, imW, imH)
            face = face_model_3d.estimate_head_pose(face, camera)
            face = face_model_3d.compute_3d_pose(face)
            face = face_model_3d.compute_face_eye_centers(face)
            face = head_pose_normalizer.normalize(imageData, face)
            cv.imshow('cvCanvas',face.normalized_image)
            console.log("face:",face)
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
    let [imW, imH]= [640, 480]
    const imageLoader = new ImageLoader(imW, imH);
    imageData = await imageLoader.getImageData('./test.jpg');
    await faceMesh.send({image: imageData})
}
Run()
