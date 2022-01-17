import os
import cv2
import timm
import torch
import urllib
import numpy as np
from torch import nn
from tqdm import tqdm
from glob import glob
from torch import optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import Config
cfg = Config()
transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, (224, 224))),
        T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                     0.225]),  # RGB
    ])
class GazeDataset(Dataset):
    """Gaze Dataset."""

    def __init__(self, root_dir, transform=T.Compose([T.ToTensor()])):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
        """
        self.images_list = glob(os.path.join(root_dir,"*.jpg"))
        self.transform = transform
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_path = self.images_list[idx]
        im_array = cv2.imread(im_path)
        im_tensor = self.transform(im_array)
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        pitch_yaw = eval(im_name[im_name.find("(")+1:im_name.find(")")])
        sample = {'im_tensor': im_tensor, 'pitch_yaw': torch.tensor(pitch_yaw)}
        return sample

gaze_dataset = GazeDataset("mpiifaces/", transform)
test_ratio = 0.25
bs = cfg.bs
lr = cfg.lr
test_len = int(len(gaze_dataset) * test_ratio)
train_subset, test_subset = torch.utils.data.random_split(
        gaze_dataset, [len(gaze_dataset)-test_len, test_len], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True, num_workers=0)
test_loader = DataLoader(test_subset, batch_size=bs, shuffle=True, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = cfg.checkpoint_path
resume_path = cfg.resume_path

net = timm.create_model("resnet18", num_classes=2)
if checkpoint_path is not None and not os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
elif resume_path is not None:
    if not os.path.isfile(resume_path):
        g = urllib.request.urlopen('https://github.com/4-geeks/xgaze-js/releases/download/v0.0.1/eth-xgaze_resnet18.pth')
        with open(resume_path, 'b+w') as f:
            f.write(g.read())
    checkpoint = torch.load(resume_path, map_location='cpu')
net.load_state_dict(checkpoint['model'])
net.to(device)
if __name__ == "__main__":
    itr = 0
    writer = SummaryWriter()
    epochs = cfg.epochs
    criterion = nn.L1Loss()
    #optimizer = optim.Adam([#{'params':net.layer3.parameters()},
    #                        {'params':net.layer4.parameters()},
    #                        {'params':net.fc.parameters()}], lr=0.0000001)#, momentum=0.9)
    optimizer = optim.Adam(net.parameters(),lr=lr)
    for e in range(epochs):
        print(f"------- epoch-{e}/{epochs} -------")
        train_loss = []
        train_bar = tqdm(train_loader)
        net.train()
        for batch in train_bar:
            inputs = batch["im_tensor"].to(device)
            labels = batch["pitch_yaw"].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),cfg.max_norm)
            optimizer.step()
            train_loss.append(loss.item())
            train_bar.set_description(f"avg_train_loss {np.mean(train_loss).item():.3f}" )
            writer.add_scalar('Loss/train', loss.item(), itr)
            itr+=1
    
        test_loss = []
        test_bar = tqdm(test_loader)
        net.eval()
        with torch.no_grad():
            for batch in test_bar:
                inputs = batch["im_tensor"].to(device)
                labels = batch["pitch_yaw"].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss.append(loss.item())
                test_bar.set_description(f"avg_test_loss {np.mean(test_loss).item():.3f}" )
                writer.add_scalar('Loss/test', np.mean(test_loss).item(), e)

torch.save({"model":net.state_dict()},checkpoint_path)
    