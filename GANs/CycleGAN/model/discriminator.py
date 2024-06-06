from .parts import *

class Discriminator(nn.Module):
    """
    Discriminator architecture is:
    c64, c128, c256, c512
    """

    def __init__(
        self, in_channels=3, features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvolutionalBlock(
                    in_channels,
                    feature,
                    kernel_size=4,
                    stride=1 if feature == features[-1] else 2,
                )
            )
            in_channels = feature
        
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial_layer(x)
