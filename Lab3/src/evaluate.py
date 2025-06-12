import torch
from utils import dice_score, dice_loss

def evaluate(model, valid_loader, device):
    model.eval()
    total_dice = 0.0
    total_loss = 0.0
    criterion = dice_loss
    
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)#predict_mask
            loss = criterion(outputs, targets)
            dice = dice_score(outputs, targets)
            
            total_loss += loss.item()
            total_dice += dice.item()
    
    num_samples = len(valid_loader)
    average_dice = total_dice / num_samples
    average_loss = total_loss / num_samples
    
    return average_dice, average_loss
