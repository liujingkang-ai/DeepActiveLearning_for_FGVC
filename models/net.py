import torch
import torch.nn as nn
import torch.nn.functional as F
from .isnet import ISNet
from .osnet import osnet
from .mgmbtnet import MGMBTNet
from .clsnet import ClsNet
from .resnet import resnet50


use_gpu = torch.cuda.is_available()
if use_gpu:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class Net(nn.Module):
    def __init__(self, n_head_small=12, n_head_big=20, num_class=100):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.object_select = osnet
        self.tf1 = MGMBTNet(n_head=n_head_small, img_size=(8, 8), in_channels=2048)
        self.tf2 = MGMBTNet(n_head=n_head_big, img_size=(16, 16), in_channels=1024)
        self.cls = ClsNet(num_class=num_class)
        self.isnet = ISNet()

    def forward(self, x):
        out_layer4, out_layer3 = self.backbone(x)

        coordinates = torch.tensor(self.object_select(out_layer3.detach()))
        batch_size = out_layer3.shape[0]
        local_imgs = torch.zeros([batch_size, 3, 256, 256]).to(DEVICE)
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(256, 256),
                                                mode='bilinear', align_corners=True)
        obj_layer4, obj_layer3 = self.backbone(local_imgs.detach())

        small_out = self.tf1(out_layer4)
        big_out = self.tf2(out_layer3)
        # print(small_out.shape, big_out.shape)
        
        cls_out = self.cls(obj_layer4, small_out, big_out)
        is_out = self.isnet(obj_layer4, small_out, big_out)
        return cls_out, is_out


# net = Net()
# x = torch.Tensor(8, 3, 256, 256)
# out1, out2 = net(x)
# print(out1.shape, out2.shape)