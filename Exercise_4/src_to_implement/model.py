import torch
import torch.nn as nn

class ResBlock(nn.Module):
    # Implementing a Residual Block that can handle different input/output channels and strides
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Creating first convolutional layer with stride for downsampling
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Creating second convolutional layer without stride
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Creating skip connection if dimensions need adjustment
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # Processing through main convolutional path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Processing through skip connection if needed
        if self.skip_connection is not None:
            identity = self.skip_connection(x)

        # Adding skip connection and applying final activation
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    # Implementing a custom ResNet architecture for binary classification
    def __init__(self):
        super().__init__()

        # Setting up initial convolution layer for feature extraction
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Adding maxpool layer for downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Creating residual blocks with increasing channels and downsampling
        self.layer1 = ResBlock(64, 64, stride=1)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 512, stride=2)

        # Adding global average pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Creating final classification layer
        self.fc = nn.Linear(512, 2)

        # Adding sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Processing through initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Processing through residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Applying global average pooling and flattening
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Generating final predictions
        x = self.fc(x)
        x = self.sigmoid(x)
        return x