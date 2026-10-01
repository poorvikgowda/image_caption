import torchvision.transforms as T

def get_transforms(split: str = "train", img_size: int = 224):
    """
    Get torchvision image transforms for image captioning.
    ImageNet normalization parameters are standard for pretrained CNNs (ResNet, VGG).
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
