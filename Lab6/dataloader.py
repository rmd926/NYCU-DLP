import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def transform_img(img):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img)
class CLEVR_DATALOADER(Dataset):
    def __init__(self, mode="train", root=None):
        super().__init__()
        assert mode in ['train', 'test', 'new_test'], "mode should be either 'train', 'test', or 'new_test'"
        
        json_root = 'C:/Users/蘇柏叡/Desktop/DL_Lab6_313553024_蘇柏叡'
        json_path = os.path.join(json_root, f'{mode}.json')
        objects_path = os.path.join(json_root, 'objects.json')

        # check if the JSON files exist
        assert os.path.exists(json_path), f"JSON file not found: {json_path}"
        assert os.path.exists(objects_path), f"Objects JSON file not found: {objects_path}"

        with open(json_path, 'r') as json_file:
            self.json_data = json.load(json_file)
            print(f"Loaded {mode} data from {json_path}")

        with open(objects_path, 'r') as json_file:
            self.objects_dict = json.load(json_file)
            print(f"Loaded objects data from {objects_path}")

        if mode == 'train':
            self.img_paths = list(self.json_data.keys())
            self.labels = list(self.json_data.values())
        else:
            self.labels = self.json_data

        self.labels_one_hot = self._generate_one_hot_labels(self.labels, self.objects_dict)
        print(f"Generated one-hot labels for {mode} data")

        self.root = root   
        self.mode = mode

    def _generate_one_hot_labels(self, labels, objects_dict):
        num_labels = len(labels)
        num_objects = len(objects_dict)
        labels_one_hot = torch.zeros(num_labels, num_objects)

        for i, label in enumerate(labels):
            indices = [objects_dict[obj] for obj in label]
            labels_one_hot[i, indices] = 1

        return labels_one_hot

    def __len__(self):
        return len(self.labels)      
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = os.path.join(self.root, self.img_paths[index])
            assert os.path.exists(img_path), f"Image not found: {img_path}"
            img = Image.open(img_path).convert('RGB')
            img = transform_img(img)
            label_one_hot = self.labels_one_hot[index]
            return img, label_one_hot
        
        elif self.mode in ['test', 'new_test']:
            label_one_hot = self.labels_one_hot[index]
            return label_one_hot

if __name__ == '__main__':
    print("Testing train mode dataset:")
    train_dataset = CLEVR_DATALOADER(root='C:/Users/蘇柏叡/Desktop/DL_Lab6_313553024_蘇柏叡/iclevr', mode='train')
    print(f"Train dataset length: {len(train_dataset)}")
    x, y = train_dataset[0]
    print(f"Image shape: {x.shape}, Label shape: {y.shape}")
    
    print("\nTesting test mode dataset:")
    test_dataset = CLEVR_DATALOADER(root='C:/Users/蘇柏叡/Desktop/DL_Lab6_313553024_蘇柏叡/iclevr', mode='test')
    print(f"Test dataset length: {len(test_dataset)}")
    y = test_dataset[0]
    print(f"Label shape: {y.shape}")
