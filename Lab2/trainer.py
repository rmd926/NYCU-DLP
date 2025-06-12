import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataloader import MIBCI2aDataset  # Ensure the data loader matches your project structure
from model.SCCNet import SCCNet        # Ensure the model path is correct
from utils import plot_metrics         # Ensure this function exists in utils.py


# Data Augmentation
def data_augmentation(data):
    # Randomly add Gaussian noise
    noise = torch.randn_like(data) * 0.02
    data = data + noise
    
    # Random time shift
    shift = np.random.randint(-5, 5)
    data = torch.roll(data, shifts=shift, dims=-1)
    
    return data
 
# Training function
def train(model, device, train_loader, valid_loader, optimizer, criterion, scheduler, epochs=300):
    best_accuracy = 0.0
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total, correct = 0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data_augmentation(data)
            optimizer.zero_grad() #清除之前的梯度
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step() #update model parameters
            
            total_loss += loss.item()
            _, predicted = output.max(1)#輸出機率最高的class
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        valid_loss, valid_accuracy = validate(model, device, valid_loader, criterion)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f},Train Acc: {train_accuracy:.2f}%, Valid Acc: {valid_accuracy:.2f}%,  LR: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            # Save the best model based on accuracy
            #save_model(model, './best_sccnet_model_SD.pth')
            save_model(model, './best_sccnet_model_LOSO.pth')
            #save_model(model, './best_sccnet_model_LOSO_FT.pth')
            print(f'Saved Best Model with Accuracy: {best_accuracy:.2f}%')

        scheduler.step(valid_loss)

    plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies)

# Validation function
def validate(model, device, valid_loader, criterion):
    model.eval()
    valid_loss, total, correct = 0, 0, 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return valid_loss / len(valid_loader), accuracy

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate datasets and data loaders
    train_dataset = MIBCI2aDataset(mode='train')
    valid_dataset = MIBCI2aDataset(mode='test')  # Assume using test set as validation set
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = SCCNet(num_classes=4, input_channels=22, Nu=22, Nt=1, dropout_rate=0.5,weight_decay=1e-4)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    optimizer = optim.Adadelta(model.parameters(),lr=1,rho=0.9, eps=1e-6)  # Use AdaDelta optimizer
    #optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print("Training Model")
    train(model, device, train_loader, valid_loader, optimizer, criterion, scheduler, epochs=300)

if __name__ == '__main__':
    main()

