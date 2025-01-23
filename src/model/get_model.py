import torchvision.models as models
import torch.nn as nn

from src.model.models.VisionModel import VisionModel

def get_model(args):

    if args.model == 'cnn':
        model = VisionModel(args)

    print(f"Model: {args.model}, Model Size: {args.model_size}, Conv Module: {args.conv_module}, FC Module: {args.fc_module}")
    print(f"Number of parameters: {count_parameters(model)}")
    print(model)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    