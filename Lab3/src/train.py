import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.resnet34_unet import ResNet34_Unet
from models.unet import UNet
from oxford_pet import load_dataset
from evaluate import evaluate
from utils import dice_score, dice_loss, plot_training_history
import time  # Import time module

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')

def train(args, train_dataset, valid_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet34_Unet().to(device) if 'resnet34' in args.model else UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    criterion = dice_loss

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print(f'Training {args.model} with epochs={args.epochs}, batch size={args.batch_size}, initial learning rate={args.lr:.4f}')

    best_dice = 0.0  # Initialize the best dice score
    start_time = time.time()  # Record the starting time

    train_loss_list = []
    valid_loss_list = []
    train_dice_list = []
    valid_dice_list = []

    best_valid_loss = float('inf')
    epochs_no_improve = 0 # Number of epochs with no improvement
    early_stop_patience = 10  # Early stopping patience

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        num_batches = len(train_loader)
        epoch_start_time = time.time()  # Start time of the current epoch

        for i, (data, mask) in enumerate(train_loader):
            inputs = data.to(device)
            masks = mask.to(device)
            optimizer.zero_grad()#clean gradient
            outputs = model(inputs)#predict_mask
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()#update 

            batch_dice = dice_score(outputs, masks).item()#calculate current batch dice score
            epoch_loss += loss.item()
            epoch_dice += batch_dice
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{num_batches} - Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}, LR: {current_lr:.1E}')

        train_loss = epoch_loss / num_batches
        train_dice = epoch_dice / num_batches
        valid_dice, valid_loss = evaluate(model, valid_loader, device)
        scheduler.step(valid_dice)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_dice_list.append(train_dice)
        valid_dice_list.append(valid_dice)

        print(f'Epoch {epoch+1} Summary - Training Loss: {train_loss:.4f}, Training Dice: {train_dice:.4f}, Validation Loss: {valid_loss:.4f}, Validation Dice: {valid_dice:.4f}')

        if valid_dice > best_dice:
            best_dice = valid_dice
            torch.save(model.state_dict(), 'saved_models/best.pth')
            print(f'Saving best model with Dice {best_dice:.4f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch {epoch+1} Duration: {epoch_duration:.2f} seconds')

        if epochs_no_improve >= early_stop_patience:
            print(f'Early stopping triggered. No improvement in validation loss for {early_stop_patience} epochs.')
            break

    return train_loss_list, valid_loss_list, train_dice_list, valid_dice_list

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument("-g", "--gpu", help="gpu id", default=0, type=int)
    #parser.add_argument("-n", "--name", help="Experiment name", default="", type=str)
    parser.add_argument('--model', '-m', type=str, default='unet', help='target model')
    parser.add_argument('--data_path', '-path', type=str, default=r'C:\Users\蘇柏叡\Desktop\DL_Lab3_313553024_蘇柏叡\dataset', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, dest='lr', help='learning rate')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_dataset = load_dataset(args.data_path, 'train')
    valid_dataset = load_dataset(args.data_path, 'valid')
    train_loss, valid_loss, train_dice, valid_dice = train(args, train_dataset, valid_dataset)
    plot_training_history(train_loss, valid_loss, train_dice, valid_dice)
 