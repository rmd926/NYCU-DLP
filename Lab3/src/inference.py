import argparse
import torch
from utils import dice_score
from oxford_pet import load_dataset
from models.resnet34_unet import ResNet34_Unet
from models.unet import UNet 
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='resnet34_unet', help='model type: resnet34_unet or unet')
    parser.add_argument('--weights', default='saved_models/best.pth', help='path to stored model weights')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument("-g", "--gpu", help="gpu id", default=0, type=int)
    return parser.parse_args()

def test(model, test_dataset, batch_size, device):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    test_dice = 0
    
    #criterion = dice_loss測試集不需要loss所以不用
    num_samples = len(test_loader)
    with torch.no_grad():
        for inputs, targets in test_loader: #inputs: image, targets: mask
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs) #predict_mask
            test_dice += dice_score(outputs, targets)
        test_dice /= num_samples
    return test_dice

if __name__ == '__main__':
    args = get_args()
    test_dataset = load_dataset(args.data_path, 'test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'resnet34_unet':
        model = ResNet34_Unet().to(device)
    elif args.model == 'unet':
        model = UNet().to(device)
    else:
        raise ValueError("Unknown model type. Please choose 'resnet34_unet' or 'unet'.")
    
    model.load_state_dict(torch.load(args.weights))
    model.to(device)
    
    test_dice = test(model, test_dataset, args.batch_size, device)
    print(f'{args.model} Testing Dice Score:{test_dice:.4f}')


