from .parts import *

class Discriminator(nn.Module):
    """
    Let Ck denote a 4 * 4 Convolution-InstanceNorm-LeakyReLU layer with
    k filters and stride 2. Discriminator architecture is: C64-C128-C256-C512.

    After the last layer, we apply a convolution to produce a 1-dimensional
    output.

    We use leaky ReLUs with a slpoe of 0.2.
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
                ConvInstanceNormLeakyReLUBlock(
                    in_channels,
                    feature,
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

        return torch.sigmoid(self.model(x))