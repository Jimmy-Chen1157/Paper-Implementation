import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downs = nn.ModuleList()
        for feature in features[0:]:
            self.downs.append(
                Block(in_channels, feature)
            )
            in_channels = feature

        self.ups = nn.ModuleList()
        in_channels_up = features[-1]
        self.initial_up = nn.ConvTranspose2d(in_channels_up, features[-2], kernel_size=2, stride=2)
        for feature in features[-2::-1]:
            self.ups.append(
                nn.Sequential(
                    Block(in_channels_up, feature),
                    nn.ConvTranspose2d(feature, feature // 2, kernel_size=2, stride=2)
                    if feature != features[0]
                    else nn.Conv2d(features[0], out_channels, kernel_size=1)
                )
            )

            in_channels_up = feature

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            if down != self.downs[-1]:
                x = self.maxpool(x)
        skip_connections = skip_connections[-2::-1]

        x = self.initial_up(x)

        for idx in range(len(self.ups)):
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = TF.resize(skip_connection, size=skip_connection.shape[2:])
            x = torch.cat([skip_connection, x], dim=1)
            x = self.ups[idx](x)
        return x


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(3, 3, 256, 256).to(device)
    model = UNet().to(device)
    pred = model(x)
    print(pred.shape)
