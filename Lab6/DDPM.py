import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import DDPMScheduler # to load the scheduler
import math
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_residual=False):
        super(ResidualConvBlock, self).__init__()
        self.is_residual = is_residual
        self.same_channels = (in_channels == out_channels)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # if the number of channels is different, use a 1x1 convolution to match the number of channels
        if not self.same_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_residual:
            residual = x if self.same_channels else self.residual_conv(x)
            return (residual + self.conv2(self.conv1(x))) / math.sqrt(2) #1.414 #use math.sqrt(2) to normalize the residual
        else:
            return self.conv2(self.conv1(x))

    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)
        return self.model(x)

class Embed(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embed, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.model(x.view(-1, self.model[0].in_features))
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # (B, N, C)
        proj_key = self.key(x).view(batch_size, -1, width * height)  # (B, C, N)
        energy = torch.bmm(proj_query, proj_key)  # (B, N, N)
        attention = F.softmax(energy, dim=-1)  # (B, N, N)
        proj_value = self.value(x).view(batch_size, -1, width * height)  # (B, C, N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels, n_feature=64, n_classes=24):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_feature = n_feature  
        self.n_classes = n_classes

        """conv first"""
        self.initial_conv = ResidualConvBlock(in_channels, n_feature, is_residual=True)
         
        """down sampling"""
        self.down1 = DownSample(n_feature, n_feature)
        self.down2 = DownSample(n_feature, 2 * n_feature)

        """self-attention layer"""
        self.attn1 = SelfAttention(n_feature)
        self.attn2 = SelfAttention(2 * n_feature)

        """bottom hidden of unet"""
        self.hidden = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        """embed time and condition"""
        self.time_embed1 = Embed(1, 2 * n_feature)
        self.time_embed2 = Embed(1, n_feature)
        self.cond_embed1 = Embed(n_classes, 2 * n_feature)
        self.cond_embed2 = Embed(n_classes, n_feature)

        self.time_embed_down1 = Embed(1, n_feature)
        self.cond_embed_down1 = Embed(n_classes, n_feature)
        self.time_embed_down2 = Embed(1, n_feature)
        self.cond_embed_down2 = Embed(n_classes, n_feature)

        """up sampling """
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feature, 2 * n_feature, 8, 8), 
            nn.GroupNorm(8, 2 * n_feature),
            nn.ReLU(True),
        )
        self.up1 = UpSample(4 * n_feature, n_feature)
        self.up2 = UpSample(2 * n_feature, n_feature)

        """output"""
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feature, n_feature, 3, 1, 1),
            nn.GroupNorm(8, n_feature),
            nn.ReLU(True),
            nn.Conv2d(n_feature, in_channels, 3, 1, 1),
        )
        self.initialize_weights()

    def initialize_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)  # usew kaiming_uniform_ to initialize the weight
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)  # set BN weight to 0.5
                m.bias.data.fill_(0.1)    # use a different constant to initialize the bias
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  #  use xavier_uniform_
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(init_fn)

    
    def forward(self, x, cond, time):
        cond_embed1 = self.cond_embed1(cond).reshape(-1, self.n_feature * 2, 1, 1) 
        time_embed1 = self.time_embed1(time).reshape(-1, self.n_feature * 2, 1, 1) 
        cond_embed2 = self.cond_embed2(cond).reshape(-1, self.n_feature, 1, 1) 
        time_embed2 = self.time_embed2(time).reshape(-1, self.n_feature, 1, 1) 
        
        cond_embed_down1 = self.cond_embed_down1(cond).reshape(-1, self.n_feature, 1, 1) 
        time_embed_down1 = self.time_embed_down1(time).reshape(-1, self.n_feature, 1, 1)
        cond_embed_down2 = self.cond_embed_down2(cond).reshape(-1, self.n_feature, 1, 1) 
        time_embed_down2 = self.time_embed_down2(time).reshape(-1, self.n_feature, 1, 1)

        x = self.initial_conv(x)  
        # down sampling + attention
        down1 = self.down1(cond_embed_down1 * x + time_embed_down1)
        down1 = self.attn1(down1)
        down2 = self.down2(cond_embed_down2 * down1 + time_embed_down2)
        down2 = self.attn2(down2)

        hidden = self.hidden(down2)

        # up sampling
        up1 = self.up0(hidden) 
        up2 = self.up1(cond_embed1 * up1 + time_embed1, down2)  
        up3 = self.up2(cond_embed2 * up2 + time_embed2, down1)
        # output
        out = self.out(torch.cat((up3, x), 1))
        return out

class DDPM(nn.Module):
    def __init__(self, unet_model, betas, noise_steps, device):
        super(DDPM, self).__init__()

        self.n_T = noise_steps
        self.device = device
        self.unet_model = unet_model

        # initialize hte DDPMScheduler
        self.scheduler = DDPMScheduler(
            beta_start=0.0001,  
            beta_end=0.02,      
            num_train_timesteps=noise_steps 
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, x, cond):
        # select a random timestep for each sample
        timestep = torch.randint(0, self.n_T, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)

        # add noise to input
        x_t = self.scheduler.add_noise(x, noise, timestep)

        # use U-Net to predict noise
        predict_noise = self.unet_model(x_t.to(self.device), cond.to(self.device), timestep.float().to(self.device) / self.n_T)

        # calculate loss
        return self.mse_loss(noise, predict_noise)

    def sample(self, cond, size):
        n_sample = len(cond)
        x_i = torch.randn(n_sample, *size).to(self.device)  # initial noise

        x_seq = [x_i]

        with tqdm(self.scheduler.timesteps, desc='Sampling Process', total=self.n_T, leave=True, ncols=100, mininterval=0.1) as pbar:
            for t in pbar:
                t = torch.full((n_sample,), t, device=self.device).long()

                # use U-Net to predict noise
                eps = self.unet_model(x_i, cond, t.float().to(self.device) / self.n_T)
                # update noise using DDPM update rule
                x_i = self.scheduler.step(eps, t[0], x_i).prev_sample
                if t[0] % (self.n_T // 10) == 0:
                    x_seq.append(x_i)
                pbar.set_description(f'Sampling: {t[0]}/{self.n_T}')

        return x_i, torch.stack(x_seq, dim=1)



# from torchsummary import summary

# model = UNet(in_channels=3, n_feature=64, n_classes=24)
# model.to('cuda' if torch.cuda.is_available() else 'cpu')

# summary(model, input_size=(3, 64, 64))
