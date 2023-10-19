import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size=3, padding=1):
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
    


class UNETBlock(nn.Module):
    def __init__(
            self, in_channels=3, init_features=64,
    ):
        super().__init__()
        features = [init_features*2**i for i in range(4)] # [64, 128, 256, 512]
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
        # self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """ Forward pass of the UNET model.
        This is differ from the original UNET paper, where the final layer is a Conv2d layer.
        We don't need a mask here since with use the output feature of this model in the Final dual unet model.
        Input:
        - x (torch.Tensor): The input tensor, with shape (batch_size, in_channels, height, width).
        Output:
        - x (torch.Tensor): The output tensor, with shape (batch_size, init_features, height, width).
        """
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

        # return self.final_conv(x)
        return x

def test_unet_block():
    print("Testing UNETBlock...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    x = torch.randn((3, 9, 64, 64)).to(device)
    model = UNETBlock(in_channels=9).to(device)
    preds = model(x)
    print(f"Shape of x: {x.shape}")
    print(f"Shape of preds: {preds.shape}")
    print("Success!")
    
    
# class UNet3D(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, init_features=32):
#         super(UNet3D, self).__init__()
#         features = [init_features*2**i for i in range(5)]
#         self.downs = nn.ModuleList()
#         self.ups = nn.ModuleList()
#         self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
#         # Down part of 3DUNet
#         for feature in features:
#             self.downs.append(Conv3dBlock(in_channels, feature, depth=6, kernel_size=3, padding=1))
#             in_channels = feature
            
#         # Up part of 3DUNet
#         for feature in reversed(features):
#             self.ups.append(
#                 nn.ConvTranspose3d(
#                     feature*2, feature, kernel_size=2, stride=2,
#                 )
#             )
#             self.ups.append(Conv3dBlock(feature*2, feature, depth=6, kernel_size=3, padding=1))
            
#         self.bottleneck = Conv3dBlock(features[-1], features[-1]*2, depth=6, kernel_size=3, padding=1)
#         self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
#     def forward(self, x):
#         skip_connections = []
        
#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)
            
#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]
        
#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)
#             skip_connection = skip_connections[idx//2]
            
#             if x.shape != skip_connection.shape:
#                 x = TF.resize(x, size=skip_connection.shape[2:])
                
#             concat_skip = torch.cat((skip_connection, x), dim=1)
#             x = self.ups[idx+1](concat_skip)
            
#         return self.final_conv(x)
        
class UNet3DBlock(nn.Module):
    """ A UNet3D model for time series data.
    """
    def __init__(self, in_channels=9, out_channels=10, ts_depth=6, init_features=64):
        """ Initialize the UNet3D model.
        Input:
        - in_channels (int): The number of input channels.
        - out_channels (int): The number of output channels.
        - ts_depth (int): The depth of the kernel. Should be the same as the depth of the input (i.e. the number of time steps in TimeSeries)
        - init_features (int): The number of initial features (Unet: 64, 128, 256, 512)
        """
        super().__init__()
        self.conv3d = Conv3dBlock(in_channels=in_channels, out_channels=32, depth=ts_depth)
        self.unet = UNETBlock(in_channels=32, init_features=init_features)
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.unet(x)
        return x
    
def test_unet3d_block():
    print("Testing UNet3DBlock...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    x = torch.randn((3, 9, 6, 64, 64)).to(device)
    model = UNet3DBlock(in_channels=9).to(device)
    preds = model(x)
    print(f"Shape of x: {x.shape}")
    print(f"Shape of preds: {preds.shape}")
    print("Success!")
    
    

class DualUNet3D(nn.Module):
    """ A Dual UNet3D model for Sentinel-1 and Sentinel-2 data TimeSeries Segmentation.
    """
    def __init__(self, s1_in_channels=2, s2_in_channels=9, out_channels=10, ts_depth=6, init_features=64, use_softmax=True):
        """ Initialize the Dual UNet3D model.
        Input:
        ---
        s1_in_channels (int): The number of input channels for Sentinel-1 data (default: 2 | VV and VH)
        s2_in_channels (int): The number of input channels for Sentinel-2 data (default: 9 | 10m bands)
        out_channels (int): Number of ouput Classes, default: 10
        ts_depth (int): The depth of the kernel. Should be the same as the depth of the input (i.e. the number of time steps in TimeSeries)
        
        """
        super().__init__()
        self.unet3d_s1 = UNet3DBlock(in_channels=s1_in_channels, out_channels=out_channels, ts_depth=ts_depth, init_features=init_features)
        self.unet3d_s2 = UNet3DBlock(in_channels=s2_in_channels, out_channels=out_channels, ts_depth=ts_depth, init_features=init_features)

        self.final_conv = nn.Conv2d(init_features*2, out_channels, kernel_size=1)
        self.use_softmax = use_softmax
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=1)
        
        
    
    def forward(self, s1_img, s2_img):
        """ Forward pass of the Dual UNet3D model.
        Input:
        ---
        s1_img (torch.Tensor): The input tensor for Sentinel-1 data, with shape (batch_size, channels, depth, height, width).
        s2_img (torch.Tensor): The input tensor for Sentinel-2 data, with shape (batch_size, channels, depth, height, width).
        
        Output:
        ---
        x (torch.Tensor): The output tensor with number of channels the same as output classes, with shape (batch_size, out_channels, height, width).
        """
        s1_feats = self.unet3d_s1(s1_img)
        s2_feats = self.unet3d_s2(s2_img)
        feats = torch.cat((s1_feats, s2_feats), dim=1)
        feats = self.final_conv(feats)
        if self.use_softmax:
            feats = self.softmax(feats)
        return feats
    
def test_dual_unet_3d():
    print("Testing DualUNet3D...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    s1_img = torch.randn((3, 2, 6, 64, 64)).to(device)
    s2_img = torch.randn((3, 9, 6, 64, 64)).to(device)
    model = DualUNet3D(s1_in_channels=2, s2_in_channels=9).to(device)
    preds = model(s1_img, s2_img)
    print(f"Shape of s1_img: {s1_img.shape}")
    print(f"Shape of s2_img: {s2_img.shape}")
    print(f"Shape of preds: {preds.shape}")
    print("Success!")

  


if __name__ == "__main__":
    test_conv3d()
    test_unet_block()
    test_unet3d_block()
    test_dual_unet_3d()
    