import torch.nn as nn

#####
# Model 1
#####

class CustomChessCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 224x224 → 224x224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 112x112

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # → 112x112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 56x56

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # → 56x56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 28x28

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # → 28x28
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # → 8x8 output for 64 squares
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # → B x (512*8*8)
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 64 * num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)         # B x 512 x 8 x 8
        x = self.classifier(x)       # B x (64 * 13)
        return x.view(-1, 64, self.num_classes)  # B x 64 x 13


#####
# Model 2 - With residual layer
#####

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample  # 1x1 conv if dimensions change

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class CustomChessCNN_v2(nn.Module):
    def __init__(self, num_classes=13):
        super(CustomChessCNN_v2, self).__init__()

        self.in_channels = 64

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Downsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Downsample
        )

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))  # Output shape: (B, 512, 8, 8)
        self.classifier = nn.Linear(512, num_classes)  # Apply per square (not globally)


    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [ResidualBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)                 # B x 512 x 8 x 8
        x = x.permute(0, 2, 3, 1)           # B x 8 x 8 x 512
        x = x.reshape(x.size(0), 64, 512)   # B x 64 x 512

        x = self.classifier(x)             # B x 64 x 13
        return x


#####
# Model 3 - With residual layer and dropout
#####

class PreActResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(x))
        if self.downsample:
            identity = self.downsample(out)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        return out + identity

class CustomChessCNN_v3(nn.Module):
    def __init__(self, in_channels=3, num_classes=13, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers (increase channels and downsample)
        self.layer1 = self._make_layer(64, 128, blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2, dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))  # output shape: (B, 512, 8, 8)
        self.classifier = nn.Linear(512, num_classes)  # for each square

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = [PreActResidualBlock(in_channels, out_channels, stride, dropout)]
        for _ in range(1, blocks):
            layers.append(PreActResidualBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)  # (B, 512, 8, 8)
        x = x.permute(0, 2, 3, 1)  # (B, 8, 8, 512)
        x = x.view(x.size(0), 64, -1)  # (B, 64, 512)
        x = self.classifier(x)  # (B, 64, 13)
        return x
