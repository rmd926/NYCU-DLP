import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack
from torch import Tensor
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from math import log10

if not os.path.exists('save_root/'):# create save_root folder for ags.save_root
    os.mkdir('save_root')

if not os.path.exists('graph/'):#save graph
    os.mkdir('graph')
    
def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD

def plot_loss(lists: list, path: str=''):
    length = list(range(1,len(lists[0])+1))
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(length, lists[0], label="train", color="blue")
    plt.plot(length, lists[1], label="valid", color="red")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.yscale("log")#將y軸轉為對數 原先使用value，但是這樣curve會變得很小，所以改成log 
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(2, 1, 2)
    plt.plot(length, lists[2], label="teacher forcing", color="green")
    plt.plot(length, lists[3], label="beta", color="black")
    plt.xlabel("Iteration")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(path)

class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.args = args
        self.type = args.kl_anneal_type
        self.ratio = args.kl_anneal_ratio
        self.stop = args.kl_anneal_stop
        self.anneal_cycle = args.kl_anneal_cycle
        self.epochs = current_epoch
        self.beta = 0.0
        if self.type == 'None':
            self.beta = 1.0
    def update(self):
        # TODO
        self.epochs += 1
        if self.type != 'None':
            self.frange_cycle_linear(0.0, self.stop, self.ratio)
        else:
            self.beta = 1.0 #如果不需要annealing，則beta設為1.0

    def get_beta(self):
        # TODO
        return self.beta
    def frange_cycle_linear(self, start, stop, ratio):
        stopped_epoch = np.ceil(self.anneal_cycle * ratio) 
        slope = (stop - start) / stopped_epoch

        if self.type == 'Monotonic': # Monotonic：只增不減，一旦達到stop保持不變
            epoch_index = self.epochs
            if epoch_index >= stopped_epoch:
                self.beta = stop
            else:
                self.beta = start + epoch_index * slope

        elif self.type == 'Cyclical': # Cyclical：週期性增減
            epoch_index = self.epochs % self.anneal_cycle 
            if epoch_index >= stopped_epoch:
                self.beta = stop
            else:
                self.beta = start + epoch_index * slope
        else:
            # None：不使用 KL annealing
            self.beta = 1.0
    
        
class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        if args.optim == 'SGD':
            self.optim      = optim.SGD(self.parameters(), lr=self.args.lr)
        elif args.optim == 'Adam':
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.00001)
        elif args.optim == 'AdamW':
            self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=0.00001)

        else:
            raise ValueError("No such optimizer")
        
        if args.lr_scheduler == 'MultiStepLR':
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.lr_milestones, gamma=args.lr_gamma)
        elif args.lr_scheduler == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=0.1, patience=5, verbose=True)
        else:
            raise ValueError("No such scheduler")
        
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        self.last_lr = args.lr
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        self.device = args.device
        self.save_graph = args.save_graph

    def update_lr(self):
        self.last_lr = [param_group['lr'] for param_group in self.optim.param_groups][0]
    
    def forward(self, img, label):
        pass

    def training_stage(self):
        min_loss = float('inf')
        max_psnr = 0
        loss_list, val_loss_list, tf_list, kl_list = [], [], [], [] 
        #adapt_TeacherForcing = random.random() < self.tfr 
        #為了增加隨機性，所以不在這裡定義，否則會一開始判斷是否<self.tfr，後面就不會再變動了
        #若放在迴圈內，每次都會重新定義，就會有機會變動
        for epoch in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            loss_temp = []
            adapt_TeacherForcing = random.random() < self.tfr
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.device)
                label = label.to(self.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                loss_temp.append(loss.detach().cpu().item())

                beta = self.kl_annealing.get_beta()
                TF_status = 'ON' if adapt_TeacherForcing else 'OFF'
                pbar.set_description(f'Epoch {epoch+1}/{self.args.num_epoch} [TeacherForcing: {TF_status} {self.tfr:.1f}, beta: {beta:.1f}, lr: {self.last_lr:.1e}]')
                pbar.set_postfix(avg_loss=np.mean(loss_temp), last_loss=loss.detach().cpu().item())

            avg_loss = np.mean(loss_temp)
            loss_list.append(avg_loss)
            kl_list.append(beta)
            tf_list.append(self.tfr)

            self.train(False)  # 設置為驗證模式
            val_loss, avg_psnr, psnr_list = self.eval()
            val_loss_list.append(val_loss)
            self.plot_psnr(psnr_list, avg_psnr, epoch)
            self.train(True)  # 設置為訓練模式

            if val_loss < min_loss or avg_psnr > max_psnr:
                min_loss = min(val_loss, min_loss)
                max_psnr = max(avg_psnr, max_psnr)
                self.save_checkpoint(val_loss, avg_psnr, epoch)

            # 學習率調整和其他更新
            self.scheduler.step(avg_psnr)
            self.update_teacher_forcing()
            self.kl_annealing.update()

            if (epoch + 1) % 10 == 0: ##預防中斷，每10個epoch存圖
                self.save_plot_loss([loss_list, val_loss_list, tf_list, kl_list])
            self.current_epoch += 1
        self.save_plot_loss([loss_list, val_loss_list, tf_list, kl_list])

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.device)
            label = label.to(self.device)
            loss, psnr_list, avg_psnr = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.cpu(), lr=self.last_lr)
        print(f'Epoch{self.current_epoch+1} Valid PSNR [max: {np.max(psnr_list):.4f}, avg: {avg_psnr:.4f}, min: {np.min(psnr_list):.4f}]')
        return loss.cpu().item(), avg_psnr, psnr_list

    def plot_psnr(self, psnr_list, avg_psnr, epoch):
        plt.close()
        plt.plot(range(len(psnr_list)), psnr_list, label=f"average PSNR: {avg_psnr:.3f}")
        plt.xlabel("Frame index")
        plt.ylabel("PSNR")
        plt.title("Per frame quality (PSNR)")
        plt.legend()
        # Construct the file path using the store_graph attribute
        file_path = os.path.join(self.save_graph, f"Ep{epoch}_PSNR.jpg")
        plt.savefig(file_path)

    def save_checkpoint(self, val_loss, avg_psnr, epoch):
        filename = f"{self.args.save_name}_{self.current_epoch+1}_best_{val_loss:.4f}_{avg_psnr:.2f}.ckpt" if self.args.save_name else f"best_{val_loss:.4f}_{avg_psnr:.2f}.ckpt"
        self.save(os.path.join(self.args.save_root, filename))

    def update_teacher_forcing(self):
        if self.current_epoch+1 >= self.tfr_sde: #因為current_epoch是從0開始，所以要+1
            self.tfr -= self.tfr_d_step
            self.tfr = max(0.0, self.tfr)

    def save_plot_loss(self, lists):
        loss_graph_stored = os.path.join(self.save_graph, f"{self.current_epoch}_LossCurve.jpg")
        plot_loss(lists, path = loss_graph_stored)
    
    def calculate_frame_loss(self, pred_img, target_img, mu, logvar, beta):
        """Calculate the reconstruction and KL loss for a single frame."""
        frame_loss = self.mse_criterion(pred_img, target_img)
        kl_loss = kl_criterion(mu, logvar, self.batch_size)
        return frame_loss + beta * kl_loss


    def train_process_frame(self, current_img, label_features):
        """Process a single frame to generate latent variables and the predicted image."""
        img_features = self.frame_transformation(current_img)#將current_img轉換成feature
        z, mu, logvar = self.Gaussian_Predictor(img_features, label_features)#透過Gaussian_Predictor得到z、mu、logvar
        fused_features = self.Decoder_Fusion(img_features, label_features, z)#fused_features把img_features和label_features合併
        pred_img = self.Generator(fused_features)#透過Generator得到pred_img
        return pred_img, mu, logvar#最後輸出pred_img、mu、logvar

    def training_one_step(self, img: Tensor, label: Tensor, adapt_TeacherForcing: bool):
        """Perform a single step of training."""
        pred_img = img[:, 0]  # Start prediction from the first frame
        total_loss = torch.zeros(1, device=self.device)
        beta = self.kl_annealing.get_beta()
        self.optim.zero_grad()

        for i in range(1, self.train_vi_len):
            # Choose the correct previous frame based on whether teacher forcing is applied
            if adapt_TeacherForcing:
                current_img = img[:, i - 1]
            else:
                current_img = pred_img            
            label_features = self.label_transformation(label[:, i]) 
            # Process the frame and compute loss
            pred_img, mu, logvar = self.train_process_frame(current_img, label_features)
            loss = self.calculate_frame_loss(pred_img, img[:, i], mu, logvar, beta)
            total_loss += loss
        
        # Average the loss over the length of the training sequence
        average_loss = total_loss / (self.train_vi_len - 1)

        if not torch.isnan(average_loss):
            average_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            self.optimizer_step()
            return average_loss
        else:
            # Handle NaN loss
            print("Encountered NaN loss, skipping update.")
            return torch.tensor(float('inf'), device=self.device)
    
    def val_one_step(self, img, label):
        # initialize the loss and PSNR list
        total_loss = torch.zeros(1, device=self.device)
        PSNR_list = []

        # use first frame as the initial prediction
        pred_img = [img[:, 0].detach()]

        for i in range(1, self.val_vi_len):
            # use the previous frame as the input
            current_img = pred_img[-1].detach()  # 防止梯度回流

            # extract features and predict the next frame
            frame_features = self.frame_transformation(current_img)
            label_features = self.label_transformation(label[:, i])

            # 這三行分別是把frame_features和label_features合併，
            # 然後透過Gaussian_Predictor得到z，最後透過Decoder_Fusion得到decoded_features
            #new_pres_img是透過Generator得到的新圖片
            z, mu, logvar = self.Gaussian_Predictor(frame_features, label_features)
            decoded_features = self.Decoder_Fusion(frame_features, label_features, z)
            new_pred_img = self.Generator(decoded_features)

            # 把新圖片加到pred_img list中
            pred_img.append(new_pred_img)

            # 計算loss
            frame_loss = self.calculate_frame_loss(new_pred_img, img[:, i], mu, logvar, self.kl_annealing.get_beta())
            total_loss += frame_loss

            # 計算PSNR並且加到PSNR_list中
            psnr_value = Generate_PSNR(new_pred_img, img[:, i]).item()
            PSNR_list.append(psnr_value)

        # calcalate the average loss and PSNR
        avg_psnr = np.mean(PSNR_list)
        avg_loss = total_loss / (self.val_vi_len - 1)

        return avg_loss, PSNR_list, avg_psnr

                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor(),
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader

  
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
      
    def tqdm_bar(self, mode, pbar: tqdm, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch+1}, lr:{lr:1.0e}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        current_lr = [param_group['lr'] for param_group in self.optim.param_groups][0]
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : current_lr,
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            #self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=0.1, patience=5, verbose=True)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']
            #self.kl_annealing.beta = 1.0 #load進來的時候要補當時的beta，未來要刪掉

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

def set_seed(seed=None):
    """if seed is None, it will generate a random seed"""
    if seed is None:
        seed = np.random.randint(0, 2**31-1)
    print(f"Using seed: {seed}")
    
    # set seed for torch, numpy and random
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    set_seed(699811706)
    os.makedirs(args.save_root, exist_ok=True)
    if args.device == 'cuda':
        args.device = torch.device('cuda', args.gpu_id)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',     type=int,   default=2)
    parser.add_argument('--lr',             type=float, default=0.001,  help="initial learning rate")
    parser.add_argument('--device',         type=str,   choices=["cuda", "cpu"],    default="cuda")
    parser.add_argument('--optim',          type=str,   choices=["Adam", "AdamW","SGD"],  default="Adam")
    parser.add_argument('--gpu',            type=int,   default=1)
    parser.add_argument('--gpu_id', '-g',   type=int,   default=0)
    parser.add_argument('--test',           action='store_true')
    parser.add_argument('--store_visualization',        action='store_true',    help="If you want to see the result while training")
    parser.add_argument('--DR',             type=str,   default=r'C:\Users\蘇柏叡\Desktop\Lab4_Dataset\Lab4_Dataset',   help="Your Dataset Path")
    parser.add_argument('--save_graph',      type=str,   default=r'graph/',      help="The path to save your pic")
    parser.add_argument('--save_root',      type=str,   default=r'save_root/',      help="The path to save your data")
    parser.add_argument('--save_name',      type=str,   default='',        help="The name(prefix) to save your data")
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--num_epoch',      type=int,   default=100,    help="number of total epoch")
    parser.add_argument('--per_save',       type=int,   default=10,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',        type=float, default=1.0,    help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',   type=int,   default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',     type=int,   default=630,    help="valdation video length")
    parser.add_argument('--frame_H',        type=int,   default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',        type=int,   default=64,     help="Width input image to be resize")
    
    # Module parameters setting
    parser.add_argument('--F_dim',          type=int,   default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',          type=int,   default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',          type=int,   default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',      type=int,   default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',            type=float,  default=0.0,   help="The initial teacher forcing ratio")#0.5
    parser.add_argument('--tfre',           type=float,  default=0.0,   help="The ending teacher forcing ratio")
    parser.add_argument('--tfr_sde',        type=int,    default=1,     help="The epoch that teacher forcing ratio start to decay")#10
    parser.add_argument('--tfr_d_step',     type=float,  default=0.1,   help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',      type=str,  default=None,    help="The path of your checkpoints")   #r'C:\Users\蘇柏叡\Desktop\Lab4_template\save_root\best_0.0099_23.17.ckpt'
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int,   default=5,      help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cycle',       help="") #Monotonic、Cyclical、None
    parser.add_argument('--kl_anneal_cycle',    type=int, default=100,               help="")#100 1.0 0.8
    parser.add_argument('--kl_anneal_stop',     type=float, default=1.0,    help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.8,              help="")
    
    # LR scheduler
    parser.add_argument('--lr_scheduler',       type=str, default='ReduceLROnPlateau',    help="")
    parser.add_argument('--lr_milestones',      type=list, default=[10, 60],           help="")
    parser.add_argument('--lr_gamma',           type=float, default=0.1,            help="")

    args = parser.parse_args()
    main(args)

#reference: about some methods、skills、存取ckpt的命名(原先因僅存為best但因為後續要寫測試時會不清楚是哪個epoch的best，所以參考了以下)
#1. https://github.com/haofuml/cyclical_annealing
#2. https://github.com/Kevin-Shih/NYCU-DLP/blob/main/Lab4/src/Trainer.py

