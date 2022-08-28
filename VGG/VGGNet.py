import torch
import torch.nn as nn

# config format (out_channels, kernel_size, stride, padding)
VGG16_config = [
    (64, 3, 1, 1),
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    (256, 3, 1, 1),
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    "M"
]

VGG11_config = [
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    "M"
]

VGG19_config = [
    (64, 3, 1, 1),
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    (256, 3, 1, 1),
    (256, 3, 1, 1),
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    (512, 3, 1, 1),
    "M"
]


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class VGG(nn.Module):

    def __init__(self, in_channels=3, num_classes=1000, architecture=VGG16_config):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture
        self.conv = self._create_conv(self.architecture)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def _create_conv(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(
                    in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3]
                )]
                in_channels = x[0]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

def test(in_channels=3, num_classes=1000, architecture=VGG16_config):
    model = VGG(in_channels, num_classes, architecture)
    x = torch.randn((2, 3, 224, 224))
    print(model(x).shape)

if __name__ == "__main__":
    test()
