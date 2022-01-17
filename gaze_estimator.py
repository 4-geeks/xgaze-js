import os
import cv2
import timm
import torch
import urllib
import numpy as np
from glob import glob
import torchvision.transforms as T
import matplotlib.pyplot as plt
from model import Model

transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, (224, 224))),
        T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                     0.225]),  # RGB
    ])
gazetr_transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, (224, 224))),
        T.ToTensor()
    ])

# gazetr_model = Model()
# state_dict = torch.load("GazeTR-H-ETH.pt",map_location="cuda")
# gazetr_model.cuda()
# gazetr_model.load_state_dict(state_dict)
# gazetr_model.eval()

checkpoint_path = "../data/eth-xgaze_resnet18.pth"
gaze_estimation_model = timm.create_model("resnet18", num_classes=2)
if not os.path.isfile(checkpoint_path):
    g = urllib.request.urlopen('https://github.com/4-geeks/xgaze-js/releases/download/v0.0.1/eth-xgaze_resnet18.pth')
    with open(checkpoint_path, 'b+w') as f:
        f.write(g.read())
checkpoint = torch.load(checkpoint_path,map_location='cpu')
gaze_estimation_model.load_state_dict(checkpoint['model'])
gaze_estimation_model.to("cuda")
gaze_estimation_model.eval()

images_list = sorted(glob("my_faces/*.jpg"))
criterion = torch.nn.L1Loss()
loss_list = []
for im_path in images_list[:300]:
    face_normalized_image = cv2.imread(im_path)
    image = gazetr_transform(face_normalized_image).unsqueeze(0)
    image = transform(face_normalized_image).unsqueeze(0)
    image = image.to("cuda")
    prediction =  gaze_estimation_model(image)
    prediction = prediction.detach().cpu().numpy()
    # prediction = gazetr_model({"face":image})
    # prediction = prediction.detach().cpu().numpy()[...,::-1]
    
    label = np.array(eval(im_path[im_path.find("(")+1:im_path.find(")")])) # * np.array([1,-1])
    loss = criterion(torch.tensor(label.copy())[1],torch.tensor(prediction.copy())[0][1]).item()
    loss_list.append(loss)
    plt.imshow(face_normalized_image)
    plt.title(f"{np.round(label,3)}, {np.round(prediction,3)[0]}")
    plt.show()
    print("label:",np.round(label,3))
    print("prediction:",np.round(prediction,3)[0])
    print("loss",loss)
    print("------------------")
print("avg loss:",np.mean(loss_list))
