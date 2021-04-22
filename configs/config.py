import sys
import os
from torchvision import transforms
import time

from utils.data_process import autoaugment, cutout

class Config():
    def __init__(self):

        self.data_path = '../data/CUB200Birds/CUB_200_2011/images/'
        self.train_data = '../data/CUB200Birds/CUB_200_2011/images/'
        self.val_data = '../data/CUB200Birds/CUB_200_2011/images/'
        self.test_data = '../data/CUB200Birds/CUB_200_2011/images/'

        self.num_class = 200
        self.label = {0 : 'NonBald', 1 : 'Bald'}

        self.use_seed = False
        self.seed = 1
        
        self.use_distributed = True
        self.device_id = [0, 1]
        self.single_device_id = '0'

        self.use_resume = False

        self.epochs = 36
        self.train_batch_size =  64
        self.val_batch_size = 16
        self.test_batch_size = 32

        self.train_per = 0.9
        self.pretrained = True
        self.is_local = False
        self.change_top = True
        self.num_worker = 4
        self.print_iter = 50

        # for PIL : height, width
        self.image_size = (256, 256)
        self.crop_image_size = (256, 256)
        
        self.train_transform_t = transforms.Compose([
            transforms.Resize(self.image_size),
            # transforms.CenterCrop(self.crop_image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            # transforms.RandomPerspective(distortion_scale=0.3),
            # transforms.RandomAffine(degrees=2, translate=(0.05, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            # autoaugment.ImageNetPolicy(),
            # transforms.TenCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.6, 1.2)),
            # cutout.Cutout(n_holes=12, length=10)
        ])
        
        self.val_transform_t = transforms.Compose([
            transforms.Resize(self.image_size),
            # transforms.CenterCrop(self.crop_image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # self.use_mixup = False
        # self.use_cutmix = False
        # self.use_fmix = False
        # self.use_ricap = False
        # self.beta = 1.0
        # self.cutmix_prob = 0.5

        self.loss_func = 'cross entropy'
        self.optim = 'sgd'
        self.loss_weight = [1] * self.num_class
        self.scheduler = 'multistep'

        self.lr = 0.01
        self.momentum_rate = 0.9
        self.weight_decay_rate = 1e-4
        self.use_nesterov = True

        self.tta_times = 12
        self.predict_by_batch = True
        self.result_path = os.path.join(sys.path[0], 'results')
        self.result_file = time.strftime('%Y-%m-%d-%H-%M', time.localtime())

        self.model_path = os.path.join(sys.path[0], 'checkpoints')
        self.log_file = 'log.txt'
