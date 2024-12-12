import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_classes=10, pretrained=False):
    # Load ResNet-18 model
    model = models.resnet18(pretrained=pretrained)
    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet50(num_classes=10, pretrained=False):
    # Load ResNet-50 model
    model = models.resnet50(pretrained=pretrained)
    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
