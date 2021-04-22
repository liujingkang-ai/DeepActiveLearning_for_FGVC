# -*- coding: utf-8 -*
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from PIL import Image, ImageFilter

from configs import config
from utils.data_process import data_loader
from models import resnet


cfg = config.Config()

if not cfg.use_distributed:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.single_device_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg.use_seed:
    os.environ['PYTHONHASHSEED'] =str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic =True


class Prediction():

    def load_model(self):

        model1 = resnest.net('resnest101', pretrained=False, is_local=False, cfg.change_top, NUM_CLASS=cfg.num_class)
        if cfg.use_distributed:
            model1 = torch.nn.DataParallel(model1, device_ids=cfg.device_id)
        cur_model = 'resnest101_epoch_35_0.98.pth'
        print(cur_model)
        model1.load_state_dict(torch.load(os.path.join(cfg.model_path, '2020-07-09-20:39', cur_model)))
        model1.eval()
        self.model1 = model1.to(device)
        print('Load model1 successful')



    def predict(self, image_path, label, test_transform):
        if cfg.use_albumentations:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(image_path).convert('RGB')

        r0 =  0
        for _ in range(cfg.tta_times):
            if cfg.use_albumentations:
                augmented = test_transform(image=img)
                tensor = augmented['image']
            else:
                tensor = test_transform(img)
            tensor = torch.unsqueeze(tensor, dim=0).float()

            v0 = self.model1(tensor.to(device))
            # t0 = self.model2(tensor.to(device))
            # r0 += v0 + t0
            r0 += v0

        val_pred = torch.max(r0, 1)[1]

        return val_pred == label


    def predict_batch(self, image_path_list, label_list, test_transform, test_data):
        r0 = 0
        for i in range(cfg.tta_times):
            if cfg.use_albumentations:
                test_dataset = data_loader.ImageDataOpenCV(image_path_list, label_list, test_transform, test_data)
            else:
                test_dataset = data_loader.ImageData(image_path_list, label_list, test_transform, test_data)
            test_loader = torch.utils.data.DataLoader(test_dataset, cfg.test_batch_size, shuffle=False, num_workers=cfg.num_worker)

            cur_res = []
            print(f'TTA: {i+1} | {cfg.tta_times}')
            for test_imgs, test_labels in tqdm(test_loader):
                test_imgs = test_imgs.to(device)
                test_labels = test_labels.to(device)
                
                p0 = self.model1(test_imgs)
                # t0 = self.model2(test_img)
                # m0 = self.model3(test_img)
                merge = p0
                # merge = p0 + t0 + m0
                cur_res.extend(merge.detach().cpu().numpy())

            cur_array = np.stack(cur_res, axis=0)
            r0 += cur_array

        test_pred = np.argmax(r0, 1)
        correct = np.sum(test_pred == np.array(label_list))

        return correct


    def get_result(self): 

        df = pd.read_csv(os.path.join(cfg.data_path, 'test.csv'))
        image_path_list = df['images'].values
        label_list = df['labels'].values
        print('Testset Size:', len(image_path_list))

        if cfg.use_albumentations:
            test_transform = cfg.train_transform_a
        else:
            test_transform = cfg.train_transform_t
        
        results = 0
        if cfg.predict_by_batch:
            results = self.predict_batch(image_path_list, label_list, test_transform, cfg.test_data)
        else:
            for i in tqdm(range(len(image_path_list))):
                results += self.predict(os.path.join(cfg.test_data, image_path_list[i]), label_list[i], test_transform)
                results = results.cpu().item()

        print('Result:', round(results/len(image_path_list), 5))



if __name__ == "__main__":
    predictor = Prediction()
    predictor.load_model()
    predictor.get_result()
