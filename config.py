class Config:
    data_root = "mpiifaces"
    resume_path = "models/eth-xgaze_resnet18.pth"
    checkpoint_path = "models/finetuned_eth-xgaze_resnet18.pth"
    bs = 128
    epochs = 50
    lr = 0.001
    max_norm = 3
    
