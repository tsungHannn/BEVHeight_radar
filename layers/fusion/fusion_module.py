import torch
import torch.nn as nn
import torch.nn.functional as F

class RCFuser(nn.Module):
    def __init__(self, in_channels):
        super(RCFuser, self).__init__()
        self.camera_conv = nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1)
        self.camera_sigmoid = nn.Sigmoid()
        self.radar_conv = nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1)
        self.radar_sigmoid = nn.Sigmoid()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.camera_maxpool = nn.MaxPool2d(1, None, 0)
        self.camera_avgpool = nn.AvgPool2d(1, None, 0)

        self.radar_maxpool = nn.MaxPool2d(1, None, 0)
        self.radar_avgpool = nn.AvgPool2d(1, None, 0)

    def forward(self, camera_features, radar_features):
        # Generate camera BEV attention
        # print("camera:", camera_features.shape) # camera: torch.Size([2, 80, 128, 128])
        # print("radar: ", radar_features.shape) # radar:  torch.Size([2, 1, 80, 128, 128]) 

        radar_features4D = torch.squeeze(radar_features, dim=1)
        # print("radar_features, ", radar_features4D.shape) # torch.Size([2, 80, 128, 128])
        # Generate camera BEV attention
        # max_pool_camera = F.max_pool2d(camera_features, kernel_size=camera_features.size()[2:])
        # avg_pool_camera = F.avg_pool2d(camera_features, kernel_size=camera_features.size()[2:])
        max_pool_camera = self.camera_maxpool(camera_features)
        avg_pool_camera = self.camera_avgpool(camera_features)

        camera_attention = self.camera_sigmoid(self.camera_conv(torch.cat([max_pool_camera, avg_pool_camera], dim=1)))

        # print(max_pool_camera.shape)
        # print(avg_pool_camera.shape)
        # print(camera_attention.shape)

        # Generate radar BEV attention
        # max_pool_radar = F.max_pool2d(radar_features4D, kernel_size=radar_features4D.size()[2:])
        # avg_pool_radar = F.avg_pool2d(radar_features4D, kernel_size=radar_features4D.size()[2:])
        max_pool_radar = self.radar_maxpool(radar_features4D)
        avg_pool_radar = self.radar_avgpool(radar_features4D)
        radar_attention = self.radar_sigmoid(self.radar_conv(torch.cat([max_pool_radar, avg_pool_radar], dim=1)))

        # print(max_pool_radar.shape)
        # print(avg_pool_radar.shape)
        # print(radar_attention.shape)

        # Apply attention weights
        fused_features = self.cbr(torch.cat([torch.mul(camera_attention, radar_features4D), torch.mul(radar_attention, camera_features)], dim=1))
        
        # print([torch.mul(camera_attention, radar_features4D).shape, torch.mul(radar_attention, camera_features).shape])
        # print(fused_features.shape)

        return fused_features