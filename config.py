class Config:
    personalize = True
    model_name = "resnet18" # "resnet18"
    pretrained = False # if True load timm pretrained model
    resume_path = "data/eth-xgaze_resnet18.pth"
    checkpoint_path = f"data/finetuned_eth-xgaze_resnet18.pth"
    personalized_path = f"data/personalized_{model_name}.pth"
    if personalize:
        data_root = "data/faces"
        lr=0.00001
    else:
        data_root = "mpiifaces"
        lr = 0.001
    bs = 128
    epochs = 400
    max_norm = 3
    
