import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optimizer, self.scheduler = self.configure_optimizers(args)
        self.prepare_training()
        self.load_checkpoint(args.checkpoint_path)
        #self.writer = SummaryWriter("logs/")
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    def train_one_epoch(self, train_loader, epoch, args):
        self.model.train()
        total_loss = 0
        total_batches = len(train_loader)
        progress = tqdm(enumerate(train_loader, start=1), total=total_batches, desc=f"Training Epoch {epoch}")
        
        for i, data in progress:
            data = data.to(args.device)
            self.optimizer.zero_grad()
            logits, targets = self.model(data)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            if i % args.accum_grad == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            running_avg_loss = total_loss / i
            progress.set_description(f"Training Epoch {epoch} - Step {i}/{total_batches}: Running Avg Loss = {running_avg_loss:.4f}")
            
        avg_loss = total_loss / total_batches
        return avg_loss

    def eval_one_epoch(self, val_loader, epoch, args):
        self.model.eval()
        total_loss = 0
        total_batches = len(val_loader)
        progress = tqdm(enumerate(val_loader, start=1), total=total_batches, desc=f"Evaluating Epoch {epoch}")
        
        with torch.no_grad():
            for i, data in progress:
                data = data.to(args.device)
                logits, targets = self.model(data)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()
                running_avg_loss = total_loss / i
                progress.set_description(f"Evaluating Epoch {epoch} - Step {i}/{total_batches}: Running Avg Loss = {running_avg_loss:.4f}")
            
        avg_loss = total_loss / total_batches
        return avg_loss

    def configure_optimizers(self, args):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        scheduler = None #torch.optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="C:/Users/蘇柏叡/Desktop/lab5_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="C:/Users/蘇柏叡/Desktop/lab5_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./transformer_checkpoints/temp/best(Ep47).pth', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=1.0)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    best_train = np.inf
    best_val = np.inf
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch, args)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch, args)
        
        if train_loss < best_train and val_loss < best_val:
            best_train = train_loss
            best_val = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/best.pth")
