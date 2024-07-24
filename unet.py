import torch 
import torch.nn as nn 
import torch.nn.functional as F

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_block(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnetBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = UnetBlock(in_channels, out_channels)

    def forward(self, x, encoder_x):
        x = self.trans_conv(x)
        x = torch.cat([x, encoder_x], dim=1)
        return self.conv_block(x)

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        self.enc_1 = UnetBlock(in_channels, features)
        self.enc_2 = Encoder(features, features * 2)
        self.enc_3 = Encoder(features * 2, features * 4)
        self.enc_4 = Encoder(features * 4, features * 8)

        self.bottleneck = Encoder(features * 8, features * 16)

        self.dec_1 = Decoder(features * 16, features * 8)
        self.dec_2 = Decoder(features * 8, features * 4)
        self.dec_3 = Decoder(features * 4, features * 2)
        self.dec_4 = Decoder(features * 2, features)
        self.last_layer = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc_1(x)
        enc2 = self.enc_2(enc1)
        enc3 = self.enc_3(enc2)
        enc4 = self.enc_4(enc3)

        bottleneck = self.bottleneck(enc4)
        
        x= self.dec_1(bottleneck, enc4)
        x= self.dec_2(x, enc3)
        x= self.dec_3(x, enc2)
        x= self.dec_4(x, enc1)
        x= torch.sigmoid(self.last_layer(x))
        return x



