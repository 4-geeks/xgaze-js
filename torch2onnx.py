import torch
import timm
import cv2
import torchvision.transforms as T

sanity_check = True
input_size = (224,224)
transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, input_size)),
        T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                     0.225]),  # RGB
    ])

checkpoint_path = "./data/eth-xgaze_resnet18.pth"
onnx_path = checkpoint_path.replace("pth","onnx")
device = "cuda" if torch.cuda.is_available() else "cpu"
gaze_estimation_model = timm.create_model("resnet18", num_classes=2)
checkpoint = torch.load(checkpoint_path,map_location='cpu')
gaze_estimation_model.load_state_dict(checkpoint['model'])
gaze_estimation_model.to(device)
gaze_estimation_model.eval()

dummy_input = torch.ones(size=(1,3,input_size[0], input_size[1])).to(device)
dummy_output = gaze_estimation_model(dummy_input)
torch.onnx.export(gaze_estimation_model, dummy_input, onnx_path, verbose=True,
                input_names=["input"], output_names=["output"],
                example_outputs=dummy_output)

if sanity_check :
    import numpy as np
    from onnxruntime import InferenceSession
    input_array =  dummy_input.detach().cpu().numpy()
    ort_sess = InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    output_array = ort_sess.run(None, {"input":input_array})
    print("output_array:",output_array[0])
    print("dummy_output:",dummy_output.detach().cpu().numpy())
