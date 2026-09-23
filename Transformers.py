import math
import torch
from torch import nn

# Rotational Positional Embedding
class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        # Register frequency function into the buffer
        self.register_buffer(
            "freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False
        )

    # Defining the function for mathematically rotating a vector by half
    def half_rotate(self, x):

        # Get the first part of the vector
        x1 = x[..., : x.shape[-1] // 2]

        # Get the second part of the vector
        x2 = x[..., x.shape[-1] // 2 :]

        # Merge and rotate the parts
        return torch.cat((-x2, x1), dim=-1)

    # Forward function for torch call
    def forward(self, length, device):

        positions = torch.arange(length, device=device)

        # Use the frequency function to rotate the positions
        freq = torch.outer(positions, self.freq)

        embedding = torch.cat(
            (freq, freq),
            dim=-1
        )

        # Return the rotations with sin and cos functions
        return (
            embedding.sin(),
            embedding.cos()
        )

    # Function that applies the rotation to QK
    def apply(self, q, k, cos, sin):

        # Extending sin and cos dimensions to match QK
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Rotate the QK
        q = (q * cos + self.half_rotate(q) * sin)
        k = (k * cos + self.half_rotate(k) * sin)

        return q, k


# Root Mean Square Normalization
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        # Make a ones mask of the tensor as a parameter
        self.weight = nn.Parameter(torch.ones(dim))

        self.eps = eps

    # Forward function for torch call
    def forward(self, x):

        # Apply the root mean square
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize and return
        return x / (rms + self.eps)

# Multi-Attention Mechanism
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.head_dim = dim // num_heads

        # Create the QKV projections
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)

        # Initiate RoPE for size of each attention head
        self.rope = RoPE(self.head_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim, bias=False)

    # Forward function for torch call
    def forward(self, x):

        # Split the shape of data
        batch_size, sequence_length, _ = x.shape

        # Project QKV
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Transpose QKV to the right shape of heads for matrx multiplication
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Get sin and cos over the length for the positional rotation
        cos, sin = self.rope(sequence_length, x.device)

        # Apply the rotation to QK
        Q, K = self.rope.apply(Q, K, cos, sin)

        # QK formula
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Get the casual mask
        casual_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        # Apply the casual mask
        scors = scores.masked_fill(casual_mask, float("inf"))

        # Attention formula
        attn = torch.softmax(scores, dim=-1) @ V
        attn = self.attn_drop(attn)

        # Reshape the attention to match the input
        attn = attn.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.dim)

        # Pass through the final layer
        return self.out(attn)


# Transformer Block
class Transformer(nn.Module):
    def __init__(self, dim, hidden, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attention = Attention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    # Forward function for torch call
    def forward(self, x):

        # Pass-Froward with residual normalization
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
