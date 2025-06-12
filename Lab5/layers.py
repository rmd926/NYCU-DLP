import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.scale = (self.head_dim ** -0.5)

        # Separate linear transformations for keys, queries, and values
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        Batch_size, num_image_tokens, dim = x.shape

        # Perform linear operations and split heads
        keys = self.key(x).reshape(Batch_size, num_image_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        queries = self.query(x).reshape(Batch_size, num_image_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.value(x).reshape(Batch_size, num_image_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention_scores, dim=-1)
        attention = self.attn_drop(attention)

        # Apply attention to values
        out = torch.matmul(attention, values)
        out = out.transpose(1, 2).reshape(Batch_size, num_image_tokens, dim)

        # Apply final linear transformation
        return self.proj(out)



class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    
    
