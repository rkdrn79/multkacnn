import torchvision.models as models
import torch.nn as nn

from src.model.models.VisionModel import VisionModel

def get_model(args):

    if args.model == 'cnn':
        model = VisionModel(args)

    print(f"Model: {args.model}")
    print(f"Number of parameters: {count_parameters(model)}")
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    