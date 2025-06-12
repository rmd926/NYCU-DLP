import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F

#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x to vqgan encoder to get the latent and zq value
    @torch.no_grad()
    def encode_to_z(self, x):
        latent, z_q, _ = self.vqgan.encode(x)
        return latent, z_q.reshape(latent.shape[0], -1)
##TODO2 step1-2:    
    
    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            def mask_func(ratio):
                return 1 - ratio
            return mask_func
        elif mode == "cosine":
            def mask_func(ratio):
                return math.cos(math.pi * ratio / 2)
            return mask_func
        elif mode == "square":
            def mask_func(ratio):
                return 1 - ratio**2
        elif mode == "cubic":
            def mask_func(ratio):
                return 1 - ratio**3
            return mask_func
        else:
            raise NotImplementedError
        
    def use_gamma(self, ratio):
        mask_ratio = self.gamma(ratio)  
        return mask_ratio
    
##TODO2 step1-3:            
    def forward(self, x):
        # Step 1: Encode input to latent space and quantize
        _, z_indices = self.encode_to_z(x)  # Assume this returns quantized indices directly
        # Step 2: Create mask - 50% chance for each token to be masked
        mask = torch.bernoulli(torch.full(z_indices.shape, 0.5, device=z_indices.device)).bool()
        # Step 3: Apply mask - replace masked tokens with the mask token ID
        masked_indices = torch.where(mask, torch.full_like(z_indices, self.mask_token_id), z_indices)
        # Step 4: Pass masked indices through the transformer
        logits = self.transformer(masked_indices)
        # Return both the logits and the original indices (ground truth for training)
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding

    @torch.no_grad()
    def inpainting(self, z_indices, mask, mask_num, ratio):
        masked_indices = mask.nonzero(as_tuple=True)
        non_masked_indices = (~mask).nonzero(as_tuple=True)

        # create a new tensor to store the indices with mask
        z_indices_with_mask = z_indices.clone()
        z_indices_with_mask[masked_indices] = self.mask_token_id
        z_indices_with_mask[non_masked_indices] = z_indices[non_masked_indices]

        # run the transformer model to get the logits
        logits = self.transformer(z_indices_with_mask)
        probs = torch.softmax(logits, dim=-1)
        # make sure the predict token is not mask token
        z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
        while torch.any(z_indices_predict == self.mask_token_id):
            z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
            
        z_indices_predict = mask * z_indices_predict + (~mask) * z_indices
        # to get prob from predict z_indices
        z_indices_predict_prob = probs.gather(-1, z_indices_predict.unsqueeze(-1)).squeeze(-1)
        z_indices_predict_prob = torch.where(mask, z_indices_predict_prob, torch.zeros_like(z_indices_predict_prob) + torch.inf)
        
        mask_ratio = MaskGit.use_gamma(self, ratio)
        mask_len = torch.floor(mask_num * mask_ratio).long()
        '''
        First, create a Gumbel distribution and sample from it. Then calculate 
        the adjusted temperature and use it to compute the confidence scores. 
        Sort these scores and find the cutoff value. Finally, update the mask
        based on this cutoff value. This process involves thresholding the confidence scores 
        to determine which elements should be masked.
        '''
        g_samples = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)
        temp_adjusted = self.choice_temperature * (1 - mask_ratio) 
        confidence_scores = z_indices_predict_prob + temp_adjusted * g_samples
        
        _, sorted_indices = torch.sort(confidence_scores, dim=-1)
        cutoff_value = confidence_scores.gather(1, sorted_indices[:, mask_len].unsqueeze(-1))

        new_mask = (confidence_scores < cutoff_value)
        return z_indices_predict, new_mask

    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


                

        
