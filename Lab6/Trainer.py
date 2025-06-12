import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from dataloader import CLEVR_DATALOADER
from torchvision.utils import save_image, make_grid
from evaluator import evaluation_model
from DDPM import DDPM, UNet

class Trainer():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.n_epoch = args.epochs
        self.run_name = args.run_name
        # Set the random seed for reproducibility
        self.set_seed(3)
        self._prepare_directories()
        self._prepare_data_loaders()
        self.build_model()
        self.evaluator = evaluation_model()

    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _prepare_directories(self):
        """Prepare directories for saving models and results."""
        os.makedirs(self.args.model_root, exist_ok=True)
        os.makedirs(self.args.result_root, exist_ok=True)

    def _prepare_data_loaders(self):
        """Prepare the data loaders for training, testing, and new testing."""
        modes = ['train', 'test', 'new_test']
        loaders = {}

        for mode in modes:
            dataset = CLEVR_DATALOADER(mode, 'C:/Users/蘇柏叡/Desktop/DL_Lab6_313553024_蘇柏叡/iclevr')
            loaders[mode] = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(mode == 'train'), num_workers=self.args.num_workers)

        # Keep these assignments outside of the function to maintain the original request
        self.train_loader = loaders['train']
        self.test_loader = loaders['test']
        self.new_test_loader = loaders['new_test']

    def build_model(self):
        """Initialize the model and optimizer."""
        unet = UNet(in_channels=3, n_feature=self.args.n_feature, n_classes=24).to(self.device)
        self.ddpm = DDPM(unet_model=unet, betas=(self.args.beta_start, self.args.beta_end), 
                         noise_steps=self.args.noise_steps, device=self.device).to(self.device)

    def train(self):
        '''
        first initialize some variables to store the best test score and new test score
        then start the training loop with tqdm to show the progress bar
        in each epoch, we first set the model to train mode, then iterate over the train_loader
        then store model weights and calculate the mean loss of the epoch
        '''
        optimizer = optim.Adam(self.ddpm.parameters(), lr=self.lr)
        best_test_epoch, best_new_test_epoch = 0, 0
        best_test_score, best_new_test_score = 0, 0

        for epoch in range(self.n_epoch):
            self.ddpm.train()
            epoch_loss = []  # store the loss of each batch
            
            with tqdm(self.train_loader, leave=True, ncols=100) as pbar:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_description(f"Epoch {epoch}, lr: {current_lr:.4e}")
                for input, cond in pbar:
                    optimizer.zero_grad()
                    input = input.to(self.device)  # [batch, 3, 64, 64]
                    cond = cond.to(self.device)  # [batch, 24]
                    loss = self.ddpm(input, cond)
                    loss.backward()
                    pbar.set_postfix_str(f"loss: {loss:.4f}")
                    optimizer.step()
                    
                    epoch_loss.append(loss.item())  
                pbar.close()
            
            mean_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch} Mean Training Loss: {mean_epoch_loss:.4f}")

            """testing time"""
            test_score, new_test_score = self.save(epoch)
            
            # update and save the best test score model
            if test_score > best_test_score:
                best_test_score = test_score
                best_test_epoch = epoch
                
            # update and save the best new test score model
            if new_test_score > best_new_test_score:
                best_new_test_score = new_test_score
                best_new_test_epoch = epoch
                
            #   save the model with test score and new test score both over 0.6
            if test_score > 0.6 and new_test_score > 0.6: 
                save_path = os.path.join(self.args.model_root, f"wanna_try_{epoch}_test{test_score:.4f}_new_test{new_test_score:.4f}.pth")
                torch.save(self.ddpm.state_dict(), save_path)
            
            # print current score and best score
            print(f"Epoch {epoch}: Test Score: {test_score:.4f}, New Test Score: {new_test_score:.4f}")
            print(f"Best Test Score: {best_test_score:.4f} (Epoch {best_test_epoch}), Best New Test Score: {best_new_test_score:.4f} (Epoch {best_new_test_epoch})")
            
            # reduce learning rate after each epoch
            # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
            #                                                 factor=0.9, patience=3, 
            #                                                 min_lr=0)
            new_lr = max(optimizer.param_groups[0]['lr'] - 2.5e-6, 0)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def save(self, epoch):
        test_score, grid1, _ = self.test(self.test_loader)
        test_new_score, grid2, _ = self.test(self.new_test_loader)
        
        path1 = os.path.join(self.args.result_root, f"test_{epoch}.png")
        path2 = os.path.join(self.args.result_root, f"test_new_{epoch}.png")
        
        save_image(grid1, path1)
        save_image(grid2, path2)
        
        return test_score, test_new_score
        

    def test(self, test_loader):
        self.ddpm.eval()
        input_gen, label = [], []
        with torch.no_grad():
            for cond in test_loader:
                cond = cond.to(self.device)
                input_fin, _ = self.ddpm.sample(cond, (3, 64, 64)) 
                input_gen.append(input_fin)
                label.append(cond)
            input_gen = torch.cat(input_gen)
            label = torch.stack(label, dim=0).squeeze()
            score = self.evaluator.eval(input_gen, label)
            grid = make_grid(input_gen, nrow=8, normalize=True)
            
            cond = torch.zeros((1, 24), device=self.device)
            # random_indices = random.sample(range(24), 6)  # 隨機選擇6個不同的索引
            # for idx in random_indices:
            #     cond[0, idx] = 1  # 將選擇的索引位置設置為1
            cond[0, 9], cond[0, 7], cond[0, 22] = 1, 1, 1  # red sphere, yellow cube, cyan cylinder
            _, input_seq = self.ddpm.sample(cond, (3, 64, 64))
            grid_seq = make_grid(input_seq, nrow=5, normalize=True, scale_each=True)
            
            # Save the final denoised image to evaluate clarity
            final_image = (input_seq[-1] + 1) / 2  # adjust the range of the image to [0, 1]
            save_image(final_image, os.path.join(self.args.result_root, "final_denoised_image.png"))
            
        return score, grid, grid_seq
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--save_path", default='ckpt')
    parser.add_argument("--run_name", default='ddpm')
    parser.add_argument("--inference", default=True, action='store_true')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    parser.add_argument('--beta_start', default=1e-4, type=float, help='start beta value')
    parser.add_argument('--beta_end', default=0.02, type=float, help='end beta value')
    parser.add_argument('--noise_steps', default=1000, type=int, help='frequency of sampling')
    parser.add_argument("--img_size", default=64, type=int, help='image size')
    parser.add_argument('--n_feature', default=64, type=int, 
                        help='time/condition embedding and feature maps dimension')#64 cuz 128 and 256 are too large for my GPU...
    parser.add_argument('--resume', default="over_60_epoch298_test0.6806_new_test0.7500.pth", action='store_true', help='resume training')
    
    parser.add_argument("--dataset_path", default="iclevr", type=str, help="root of dataset dir")
    parser.add_argument("--model_root", default="ckpt", type=str, help="model ckpt path")
    parser.add_argument("--result_root", default="results", type=str, help="save img path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)

    #args.inference = True
    if args.inference:
        print("Running inference...")
        
        path = os.path.join(args.model_root, "DL_lab6_313553024_蘇柏叡.pth")
        #
        trainer.ddpm.load_state_dict(torch.load(path))
        test_score, grid_test, grid_process = trainer.test(trainer.test_loader)
        # Save test images
        path = os.path.join(args.result_root, "___test___.png")
        
        save_image(grid_test, path)
        #denoise_process(trainer.ddpm, trainer.test_loader, args, save_prefix='test')
        

        test_new_score, grid_test_new, _ = trainer.test(trainer.new_test_loader)
        # Save new test images
        path = os.path.join(args.result_root, "___test_new___.png")
        save_image(grid_test_new, path)
        #denoise_process(trainer.ddpm, trainer.new_test_loader, args, save_prefix='new_test')

        print("Test accuracy: {:.4f}, New test accuracy: {:.4f}".format(test_score, test_new_score))
        print("Inference done")
    elif not args.inference:
        if args.resume:
            print("Resume training...")
            path = os.path.join(args.model_root, "over_60_epoch298_test0.6806_new_test0.7500.pth")
            trainer.ddpm.load_state_dict(torch.load(path))
        print("Start training...")
        trainer.train()  

