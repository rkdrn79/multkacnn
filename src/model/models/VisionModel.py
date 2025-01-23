import torch 
from torch import nn

from src.model.layers.KANConv import KAN_Convolutional_Layer
from src.model.layers.KANLinear import KANLinear
from src.model.layers.MultKANConv import MultKAN_Convolutional_Layer
from src.model.layers.MultKANLinear import MultKANLinear

import pdb

class VisionModel(nn.Module):
    def __init__(self, args):
        super(VisionModel, self).__init__()
        self.args = args
        self.model_size = self.args.model_size
        self.num_classes = self.args.num_classes

        self.grid_size = self.args.grid_size
        
        if self.model_size == "small":
            self.out_channels = 5
            self.fc_size = 125
        
        elif self.model_size == "medium":
            self.out_channels = 10
            self.fc_size = 250
        
        elif self.model_size == "large":
            self.out_channels = 10
            self.fc_size = 250            

        if args.conv_module == "cnn":
            self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=(0,0))
            self.conv2 = nn.Conv2d(5, self.out_channels, kernel_size=3, padding=(0,0))

        elif args.conv_module == "kan":
            self.conv1 = KAN_Convolutional_Layer(in_channels=1,
                out_channels= 5,
                kernel_size= (3,3),
                grid_size = self.grid_size,
                padding =(0,0)
            )

            self.conv2 = KAN_Convolutional_Layer(in_channels=5,
                out_channels= self.out_channels,
                kernel_size = (3,3),
                grid_size = self.grid_size,
                padding =(0,0)
            )

        elif args.conv_module == 'multkan':
            self.conv1 = MultKAN_Convolutional_Layer(in_channels=1,
                out_channels= 5,
                kernel_size= (3,3),
                grid_size = self.grid_size,
                padding =(0,0)
            )

            self.conv2 = MultKAN_Convolutional_Layer(in_channels=5,
                out_channels= self.out_channels,
                kernel_size = (3,3),
                grid_size = self.grid_size,
                padding =(0,0)
            )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        if args.fc_module == "linear":
            self.fc1 = nn.Linear(self.fc_size, self.num_classes)

        elif args.fc_module == "kan":
            self.fc1 = KANLinear(
            self.fc_size,
            self.num_classes,
            grid_size=self.grid_size,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
            )
        
        elif args.fc_module == "multkan":
            self.fc1 = MultKANLinear(
                self.fc_size,
                self.num_classes,
                grid_size=self.grid_size,
                spline_order=3,
                scale_noise=0.01,
                scale_base=1,
                scale_spline=1,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[0,1],
            )

    def forward(self, x):
        x = self.conv1(x)
        if self.args.conv_module == 'cnn':
            x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        if self.args.conv_module == 'cnn':
            x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.log_softmax(x, dim=1)

        return x

