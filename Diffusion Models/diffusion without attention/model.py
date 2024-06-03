from parts import *

class DiffusionModel(nn.Module):
    def __init__(self, T, embed_size, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.T = T
        self.embed_size = embed_size
        self.n_channels = n_channels

        self.dim = 4
        self.conv1 = nn.Conv2d(n_channels, self.dim, kernel_size=3, padding=1)

        self.time_encoding = TimeEncoding(T, embed_size)
        self.projection = nn.Linear(embed_size, self.dim)

        self.inc = (DoubleConv(self.dim, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, z, time_input):
        time_enc = self.time_encoding(time_input)  # (batch_size, embed_size)
        time = self.projection(time_enc)  # (batch_size, self.dim)

        # add time data to every pixel
        z = z.permute(0, 3, 1, 2)
        z = self.conv1(z)
        z = time.unsqueeze(1).unsqueeze(1).permute(0, 3, 1, 2) + z  # (batch_size, )
        # z = z.permute(0, 3, 1, 2)
        x1 = self.inc(z)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
