import torch
import torch.nn as nn

class LightweightUNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(LightweightUNet, self).__init__()

        # Encoder
        self.encoder1 = self.encoder_block(input_channels, 32)
        self.encoder2 = self.encoder_block(32, 64)
        self.encoder3 = self.encoder_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.decoder3 = self.decoder_block(256, 128)
        self.decoder2 = self.decoder_block(128, 64)
        self.decoder1 = self.decoder_block(64, 32)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_block(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Bottleneck
        x = self.bottleneck(x3)

        # Decoder
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)

        # Classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
