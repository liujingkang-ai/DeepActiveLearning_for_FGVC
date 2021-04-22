import os
import numpy as np

from PIL import Image, ImageFilter
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

from configs import config
from torch.utils.data import Dataset

cfg = config.Config()


class ImageData(Dataset):
    def __init__(self, path_list, label_list, transform, image_path):
        self.dataset = path_list
        self.label = label_list
        self.transform = transform
        self.image_path = image_path

    def __getitem__(self, item):
        imgname = self.dataset[item]
        label = self.label[item]

        image = Image.open(os.path.join(self.image_path, str(imgname))).convert('RGB')
        if self.transform is not None:
            img = self.transform(image)

        return img, np.array(label)

    def __len__(self):
        return len(self.dataset)


class TestImageData(Dataset):
    def __init__(self, path_list, transform, image_path):

        self.dataset = path_list
        self.transform = transform
        self.image_path = image_path


    def __getitem__(self, item):
        imgname = self.dataset[item]

        image = Image.open(os.path.join(self.image_path, str(imgname))).convert('RGB')

        if self.transform is not None:
            img = self.transform(image)
            
        return img, imgname

    def __len__(self):
        return len(self.dataset)