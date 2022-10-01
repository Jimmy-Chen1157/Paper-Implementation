import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, img_channels, d_features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                img_channels, d_features, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(d_features, d_features*2, 4, 2, 1),
            self._block(d_features*2, d_features*4, 4, 2, 1),
            self._block(d_features*4, d_features*8, 4, 2, 1),
            nn.Conv2d(d_features*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
                ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        return block

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):

    def __init__(self, z_dim, img_channels, g_features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, g_features*16, 4, 1, 0),
            self._block(g_features*16, g_features*8, 4, 2, 1),
            self._block(g_features*8, g_features*4, 4, 2, 1),
            self._block(g_features*4, g_features*2, 4, 2, 1),
            nn.ConvTranspose2d(
                g_features*2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):

        block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        return block

    def forward(self, x):
        return self.gen(x)

def test():
    N, in_channels, H, W = 1, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    print(disc(x).size())
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    print(gen(z).size())

if __name__ == "__main__":
    test()
