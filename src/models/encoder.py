import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    """
    Pretrained CNN Feature Extractor (Encoder).
    Extracts spatial feature maps (batch_size, num_pixels, feature_dim) from input images.
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        embed_size: int = 256,
        freeze_encoder: bool = True
    ):
        super(CNNEncoder, self).__init__()
        self.backbone_name = backbone.lower()
        self.freeze_encoder = freeze_encoder

        if self.backbone_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            modules = list(resnet.children())[:-2]  # Remove avgpool and fc
            self.cnn = nn.Sequential(*modules)
            self.encoder_dim = 2048
        elif self.backbone_name == "resnet101":
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            modules = list(resnet.children())[:-2]
            self.cnn = nn.Sequential(*modules)
            self.encoder_dim = 2048
        elif self.backbone_name == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.cnn = vgg.features
            self.encoder_dim = 512
        else:
            raise ValueError(f"Unsupported CNN backbone: {backbone}")

        # Linear layer to transform visual feature dimension if needed
        self.fine_tune(freeze_encoder)

    def fine_tune(self, freeze: bool = True):
        """
        Freeze or unfreeze CNN parameters for fine-tuning.
        """
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input: images of shape (batch_size, 3, H, W)
        Output: spatial features of shape (batch_size, num_pixels, encoder_dim)
                e.g. (batch_size, 49, 2048) for 224x224 input on ResNet50
        """
        features = self.cnn(images)  # (batch_size, encoder_dim, H', W')
        batch_size, channels, h, w = features.shape

        # Reshape to (batch_size, H'*W', channels)
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size, h * w, channels)
        return features
