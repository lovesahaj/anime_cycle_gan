from django.db import models

# Create your models here.
import torch
import torch.nn as nn

from .config import DEVICE


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(ConvBlock, self).__init__()
        blocks = list()

        blocks.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode='reflect',
            ))
        blocks.append(nn.InstanceNorm2d(out_channels))
        blocks.append(nn.ReLU())

        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)


class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockUp, self).__init__()

        block = list()

        block.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ))
        block.append(nn.InstanceNorm2d(out_channels))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect',
            stride=1,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        no_of_features=64,
        num_residual=6,
        no_of_downsamples=2,
    ):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels,
                no_of_features,
                kernel_size=7,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(no_of_features),
            nn.ReLU(),
        )

        downsample = list()
        residuals = list()
        upsample = list()

        for i in range(no_of_downsamples):
            new_in = no_of_features * (no_of_downsamples**i)
            new_out = no_of_features * (no_of_downsamples**(i + 1))
            downsample.append(
                ConvBlock(new_in, new_out, kernel_size=3, padding=1, stride=2))

        no_of_features = new_out

        for i in range(num_residual):
            residuals.append(ResBlock(no_of_features, no_of_features))

        for i in range(no_of_downsamples):
            new_in = no_of_features // (no_of_downsamples**i)
            new_out = no_of_features // (no_of_downsamples**(i + 1))
            upsample.append(ConvBlockUp(new_in, new_out))

        no_of_features = new_out

        self.downsample = nn.Sequential(*downsample)
        self.residual = nn.Sequential(*residuals)
        self.upsample = nn.Sequential(*upsample)

        self.final = nn.Sequential(
            nn.Conv2d(
                no_of_features,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode='reflect',
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.initial(x)
        out = self.downsample(out)
        out = self.residual(out)
        out = self.upsample(out)
        return self.final(out)

    def save(self, optimizer=None, name_file='generator.pth.tar'):
        if optimizer is None:
            checkpoint = {'generator': self.state_dict(), 'optimizer': None}
        else:
            checkpoint = {
                'generator': self.state_dict(),
                'optimizer': optimizer.state_dict()
            }

        torch.save(checkpoint, name_file)

    def load(self, optimizer=None, name_file='generator.pth.tar'):
        c = torch.load(name_file, map_location=DEVICE)

        if optimizer is not None:
            self.load_state_dict(c['generator'])
            optimizer.load_state_dict(c['optimizer'])
        else:
            self.load_state_dict(c['generator'])


class Critic(nn.Module):
    def __init__(self, in_channels, num_layers, no_of_features):
        super(Critic, self).__init__()

        block = [
            ConvBlock(
                in_channels,
                no_of_features,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        ]

        for i in range(0, num_layers - 1):
            new_in = no_of_features * (2**i)
            new_out = no_of_features * (2**(i + 1))

            block.append(
                ConvBlock(
                    new_in,
                    new_out,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ))

        block.append(nn.Conv2d(new_out, 1, 4, 1, 0))

        self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.model(x)

    def save(self, optimizer=None, name_file='critic.pth.tar'):
        if optimizer is None:
            checkpoint = {'critic': self.state_dict(), 'optimizer': None}
        else:
            checkpoint = {
                'critic': self.state_dict(),
                'optimizer': optimizer.state_dict()
            }

        torch.save(checkpoint, name_file)

    def load(self, optimizer=None, name_file='critic.pth.tar'):
        c = torch.load(name_file, map_location=DEVICE)

        if optimizer is not None:
            self.load_state_dict(c['critic'])
            optimizer.load_state_dict(c['optimizer'])
        else:
            self.load_state_dict(c['critic'])


if __name__ == "__main__":
    print(torch.__version__)
    # print(ConvBlock(3, 64))
    # print(ResBlock(256, 256))

    img = torch.randn((1, 3, 128, 128)).cuda()
    gen = Generator(3, 3).cuda()

    # print(gen(img).shape)

    # disc = Critic(3, 4, 64).cuda()
    print(gen(gen(img)).shape)