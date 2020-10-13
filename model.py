import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int):
        super(Generator, self).__init__()

        self.noise_layer = nn.Linear(noise_dim, 16 * 8 * 8)
        self.noise_activation = nn.SELU(inplace=True)
        self.conv1 = self.__conv_block(16, 64)
        self.conv2 = self.__conv_block(64, 128)
        self.conv3 = self.__conv_block(128, 256)
        self.output_layer = nn.Conv2d(256, 3, 3, padding=1)

    @staticmethod
    def __conv_block(in_channels: int, out_channels: int):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                             nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.BatchNorm2d(out_channels, momentum=.8),
                             nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.noise_activation(self.noise_layer(x))
        h = h.view(-1, 16, 8, 8)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.output_layer(h)
        return torch.tanh(h)


class Discriminator(nn.Module):
    def __init__(self, image_channel: int):
        super(Discriminator, self).__init__()

        self.conv1 = self.__conv_block(image_channel, 128)
        self.conv2 = self.__conv_block(128, 64)
        self.conv3 = self.__conv_block(64, 32)
        self.conv4 = self.__conv_block(32, 16)
        self.output_layer = nn.Linear(16 * 3 * 3, 1)

    @staticmethod
    def __conv_block(in_channels: int, out_channels: int, dropout: float = .0):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=2),
                             nn.LeakyReLU(0.2, inplace=True),
                             nn.Dropout2d(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = torch.flatten(h, 1)
        h = self.output_layer(h)
        return torch.sigmoid(h)
