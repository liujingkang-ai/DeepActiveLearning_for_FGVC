import torch
import torch.nn as nn
import torch.nn.functional as F


class ISNet(nn.Module):
    def __init__(self, inner_dim=128, object_dim=2048, kernel_size=8, feat_dim=512, num_class=1):
        super().__init__()
        self.gap = nn.AvgPool2d(kernel_size=kernel_size)
        self.fc1 = nn.Linear(object_dim, inner_dim)
        self.fc2 = nn.Linear(feat_dim, inner_dim)
        self.fc3 = nn.Linear(feat_dim, inner_dim)
        
        # self.cls = nn.Linear(inner_dim*(13+21+1), num_class)
        self.cls = nn.Sequential(
            nn.BatchNorm1d(inner_dim*(1+13+21)),
            nn.Linear(inner_dim*(1+13+21), num_class * 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_class * 16),
            nn.Linear(num_class * 16, num_class)
        )
    
    def forward(self, object_level, att1, att2):
        obj_level = F.relu(self.fc1(self.gap(object_level).view(object_level.shape[0], -1)), inplace=True)
        att1_level = F.relu(self.fc2(att1)).view(att1.shape[0], -1)
        att2_level = F.relu(self.fc3(att2)).view(att2.shape[0], -1)
        # print(obj_level.shape, att1_level.shape, att2_level.shape)
        out = self.cls(torch.cat([obj_level, att1_level, att2_level], dim=-1))
        return out