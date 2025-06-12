import torch
import numpy as np
import os
from torch.utils.data import Dataset

class MIBCI2aDataset(Dataset):
    def _getFeatures(self, filePath):
        features = []
        for file_name in sorted(os.listdir(filePath)):
            file_path = os.path.join(filePath, file_name)
            data = np.load(file_path)
            features.append(data)
        return np.concatenate(features, axis=0)

    def _getLabels(self, filePath):
        labels = []
        for file_name in sorted(os.listdir(filePath)):
            file_path = os.path.join(filePath, file_name)
            data = np.load(file_path)
            labels.append(data)
        return np.concatenate(labels, axis=0)

    def __init__(self, mode):
        assert mode in ['train', 'test', 'finetune'], "Mode must be 'train', 'test', or 'finetune'"
        base_path = 'C:/Users/蘇柏叡/Desktop/DLP_Lab2_313553024_蘇柏叡/lab2/dataset/'  # 修改为您的实际路径
        if mode == 'train':
            feature_path = os.path.join(base_path, 'SD_train/features/') #option 1
            label_path = os.path.join(base_path, 'SD_train/labels/')
            # feature_path = os.path.join(base_path, 'LOSO_train/features/') #option 2
            # label_path = os.path.join(base_path, 'LOSO_train/labels/')
            # feature_path = os.path.join(base_path, 'FT/features/') #option 3
            # label_path = os.path.join(base_path, 'FT/labels/')
        elif mode == 'finetune':
            feature_path = os.path.join(base_path, 'FT/features/')
            label_path = os.path.join(base_path, 'FT/labels/')
        elif mode == 'test':
            feature_path = os.path.join(base_path, 'SD_test/features/') #option 1
            label_path = os.path.join(base_path, 'SD_test/labels/')
            # feature_path = os.path.join(base_path, 'LOSO_test/features/') #option 2、3
            # label_path = os.path.join(base_path, 'LOSO_test/labels/')
        self.features = self._getFeatures(feature_path)
        self.labels = self._getLabels(label_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if len(feature.shape) == 2:  
            feature = feature.reshape(1, *feature.shape)
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
