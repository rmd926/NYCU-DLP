import torch
from torch.utils.data import Dataset as torchData
from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
import os
from glob import glob

def get_key(fp):
    filename = os.path.basename(fp)  
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

class Dataset_Dance(torchData):
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'"
        
        folder = 'train' if mode == 'train' else 'val'
        self.img_folder = sorted(glob(os.path.join(root, folder, folder + '_img/*.png')), key=get_key)
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        imgs = []
        labels = []
        for i in range(self.video_len):
            img_path = self.img_folder[index * self.video_len + i]
            label_path = img_path.replace('_img', '_label')
            
            imgs.append(self.transform(imgloader(img_path)))
            labels.append(self.transform(imgloader(label_path)))

        return stack(imgs), stack(labels)
