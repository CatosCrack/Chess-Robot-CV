import numpy as np
from torch import nn


class Model():
    def __init__(self) -> None:
        
        # Feature extractor with convolutional layers
        # TODO: Update kernel, stride, and padding once image size is determined
        self.features = nn.Sequential(
            # Uses padding=2 and stride=1 to preserve dimensions
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=2, stride=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces spatial dimensions by half
            
            # Uses padding=1 and stride=1 to preserve even dimensions
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduces spatial dimensions by half
        )

        # Classifier with fully connected layers
        self.classifier = nn.Sequential()

    def forward(self, x):
        pass

    def stack_images(self, images):
        pass