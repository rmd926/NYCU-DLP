
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

def dice_score(pred_mask: Tensor, gt_mask: Tensor) -> Tensor:
    with torch.no_grad():
        # 二值化預測遮罩
        pred_mask_bin = torch.where(pred_mask >= 0.5, torch.tensor(1.0, device=pred_mask.device), torch.tensor(0.0, device=pred_mask.device))
        smooth = 1e-6
        
        # 利用矩陣乘法來計算交集
        inter = (pred_mask_bin * gt_mask).sum()
        
        # 利用矩陣操作計算聯集
        union = pred_mask_bin.sum() + gt_mask.sum()
        
        # 計算 Dice 系數
        dice = (2. * inter + smooth) / (union + smooth)
        
        return dice

def dice_loss(pred_mask: Tensor, gt_mask: Tensor) -> Tensor:
    # 計算 Dice 系數
    inter = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    smooth = 1e-6
    dice = (2. * inter + smooth) / (union + smooth)

    # Dice loss 為 1 減 Dice 系數
    return 1 - dice


def plot_training_history(train_loss, valid_loss, train_dice, valid_dice):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, valid_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dice, 'bo-', label='Training Dice')
    plt.plot(epochs, valid_dice, 'ro-', label='Validation Dice')
    plt.title('Training and Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

