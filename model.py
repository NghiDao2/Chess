import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        # returns (x - μ)/sqrt(σ^2 + epsilon)
        # epsilon = small number like 1e-5 ensures the value returned does not blow up


class SelfAttention(nn.Module):

    def __init__(self, n_embed, n_head, dropout=0, bias=False):
        super().__init__()

        # self.attention contains the Q, K, V weights
        self.attention = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        self.projection = nn.Linear(n_embed, n_embed)
        self.residual_dropout = nn.Dropout(p=dropout)

        self.n_embed = n_embed
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias



    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.attention(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (n_batch, num heads, n_seq, head size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (n_batch, num heads, n_seq, head size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (n_batch, num heads, n_seq, head size)

        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        
        output = output.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        output = self.projection(output)
        output = self.residual_dropout(output)

        return output


class MLP(nn.Module):
    
    def __init__(self, n_embed, dropout=0, bias=False):
        super().__init__()
        self.up_sample = nn.Linear(n_embed, 4 * n_embed, bias=bias)
        self.gelu = nn.GELU()
        self.down_sample = nn.Linear(4 * n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.up_sample(x)
        output = self.gelu(output)
        output = self.down_sample(output)
        output = self.dropout(output)
        return output


class Block(nn.Module):

    def __init__(self, n_embed, n_head, dropout=0, bias=False):
        super().__init__()
        self.layer_norm_1 = LayerNorm(n_embed, bias)
        self.attention = SelfAttention(n_embed, n_head, dropout, bias)
        self.layer_norm_2 = LayerNorm(n_embed, bias)
        self.mlp = MLP(n_embed, dropout, bias)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x


class ChessModelConfig:
    n_layer: int = 6
    n_head: int = 8
    n_embed: int = 64
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms.

class ChessModel(nn.Module):
    def __init__(self, config : ChessModelConfig):
        super().__init__()
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        self.bias = config.bias

        # Embedding layers
        self.piece_embed = nn.Embedding(16, self.n_embed)
        self.position_embed = nn.Embedding(64, self.n_embed)
        self.castling_embed = nn.Embedding(4, self.n_embed)
        self.enpassant_embed = nn.Embedding(1, self.n_embed)
        self.repetition_embed = nn.Embedding(3, self.n_embed)
        self.rule50_embed = nn.Embedding(50, self.n_embed)

        with torch.no_grad():
            self.piece_embed.weight[torch.tensor([6, 7, 14, 15])] = 0
        self.piece_embed.weight.requires_grad = True 

        self.blocks = nn.ModuleList([Block(self.n_embed, self.n_head, self.dropout, self.bias) for _ in range(self.n_layer)])
        self.layer_norm = LayerNorm(self.n_embed, self.bias)
        self.policy = nn.Linear(self.n_embed, 1882, bias=self.bias)
        self.evaluation = nn.Linear(self.n_embed, 1, bias=self.bias)



    def forward(self, x): # assumes this input is already in the correct format
        
        for block in self.blocks:
            x = block(x)
            
        x = self.layer_norm(x) 
        x = x.transpose(1, 2)  # Change to [batch_size, n_emb, ith chess piece]
        x = torch.mean(x, dim=-1)  # Mean to [batch_size, n_emb, 1] compress all the representation into a single one
        x = x.squeeze(-1)  # Remove last dimension [batch_size, n_emb]
        move_logits = self.policy(x)
        evaluation = self.evaluation(x)

        return [evaluation, move_logits]


    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

