import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, padding):
        """
        A 3D convolutional block that processes time series data.

        Args:
        - in_channels (int): The number of input channels.
        - out_channels (int): The number of output channels.
        - depth (int): The depth of the kernel. Should be the same as the depth of the input (i.e. the number of time steps in TimeSeries)
        - kernel_size (int or tuple): The size of the kernel.
        - padding (int or tuple): The size of the padding.

        Returns:
        - x (torch.Tensor): The output tensor.
        """
        super().__init__()
        self.depth = depth
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # Remove Depth
            nn.Conv3d(out_channels, out_channels, kernel_size=(depth, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        """
        Forward pass of the 3D convolutional block.

        Args:
        - x (torch.Tensor): The input tensor, with shpae (batch_size, channels, depth, height, width).

        Returns:
        - x (torch.Tensor): The output tensorl, with shape (batch_size, channels, height, width).
        """
        assert self.depth == x.shape[2], f"Depth of input tensor ({x.shape[2]}) does not match initialized depth ({self.depth})"
        
        x = self.conv(x)
        x = x.squeeze(2) # Remove Depth which is 1
        return x
    
    
def test_conv3d():
    print("Testing Conv3d...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    x = torch.randn((1, 9, 6, 64, 64)).to(device) # (batch_size, channels, depth, height, width)
    model = Conv3dBlock(in_channels=9, out_channels=32,depth=6, kernel_size=3, padding=1).to(device)
    preds = model(x)
    
    print(f"Shape of x: {x.shape}")
    print(f"Shape of preds: {preds.shape}")
    print("Success!")



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    x = torch.randn((3, 1, 64, 64)).to(device)
    model = UNET(in_channels=1, out_channels=1).to(device)
    preds = model(x)
    assert preds.shape == x.shape
    print("Success!")

if __name__ == "__main__":
    test_conv3d()
    test()