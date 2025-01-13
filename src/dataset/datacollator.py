import torch

class VisionDataCollator:
    def __init__(self, args):
        self.args = args

    def __call__(self, features):
        inputs = [torch.tensor(feature[0]) for feature in features]
        inputs = torch.stack(inputs).to(torch.bfloat16)

        labels = [torch.tensor(feature[1]) for feature in features]
        labels = torch.stack(labels).to(torch.bfloat16)

        return {
            'inputs': inputs,
            'labels': labels
        }