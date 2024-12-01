import torch
import torch.nn as nn
import torch.nn.functional as F
from tnet import TNet


class PointNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, spatial_dims=2
    ):  # in=(x,y,p,vx,vy), out=(p,vx,vy)
        super().__init__()
        self.spatial_dims = spatial_dims
        self.transform1 = TNet(k=spatial_dims)  # Only transform spatial coordinates
        self.transform2 = TNet(k=64)

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Regression head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # Split spatial and feature coordinates
        coords = x[:, : self.spatial_dims, :]  # Only x,y coordinates
        features = x[:, self.spatial_dims :, :]  # p,vx,vy

        # Transform spatial coordinates
        coords_transform = self.transform1(coords)
        coords = torch.bmm(coords.transpose(2, 1), coords_transform).transpose(2, 1)

        # Concatenate transformed coordinates with features
        x = torch.cat([coords, features], dim=1)

        # Continue with regular PointNet processing
        x = F.relu(self.bn1(self.conv1(x)))

        features_transform = self.transform2(x)
        x = torch.bmm(x.transpose(2, 1), features_transform).transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x
