import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=10, pretrained=True):
    """
    Returns a MobileNetV2 model with a modified classifier for num_classes.
    """
    # Load pre-trained MobileNetV2
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    # Modify the classifier
    # The original classifier is:
    # (classifier): Sequential(
    #   (0): Dropout(p=0.2, inplace=False)
    #   (1): Linear(in_features=1280, out_features=1000, bias=True)
    # )
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model
