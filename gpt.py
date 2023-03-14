import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat


class SwiGLU(nn.Module):
  def forward(self, x):
    x, g = x.chunk(2, dim=-1)
    return x * F.silu(g)


class RMSNorm(nn.Module):
  def __init__(self, d):
    super().__init__()
    assert isinstance(d, int), 'd must be an int'
    self.g = nn.Parameter(torch.ones(d))

  def forward(self, x):
    assert x.shape[-1] == self.d, f'd must be {self.d}'
    return self.g * x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)


class RotaryEmbedding(nn.Module):
  def __init__(self, l, d):
    super().__init__()
    assert d % 2 == 0, f'd must be even'
    self.d, self.l = d, l
    θ = 1e4 ** (repeat(-torch.arange(0, d, 2), 'i -> (i j)', j=2) / d)
    lθ = einsum('i, j -> ij', torch.arange(l), θ)
    self.sin = nn.Parameter(lθ.sin(), requires_grad=False)
    self.cos = nn.Parameter(lθ.cos(), requires_grad=False)

  def forward(self, x):
    _, l, d = x.shape
    flip = torch.cat((-x[..., 1::2, None], x[..., ::2, None]), -1)
    x_ = rearrange(flip, 'b l x y -> b l (x y)')
    return x * self.cos[:l] + x_ * self.sin[:l]


class Layer(nn.Module):
  def __init__(self, d, nh):
    super().__init__()
    assert d % nh == 0, 'nh must divide d'
    self.d, self.c = d, d ** -0.5

    self.wx = nn.Linear(d, 3 * d, bias=False)
    self.wo = nn.Linear(d, d, bias=False)
    self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
    self.ff = nn.Sequential(nn.Linear(d, 6 * d), SwiGLU(), nn.Linear(3 * d, d))

    self.rotate = lambda x: x

    self.split = lambda x: rearrange(
      rearrange(x, 'b l d -> b d l'), 'b (nh dh) l -> b nh dh l', nh=nh
    )
    self.concat = lambda x: rearrange(
      rearrange(x, 'b nh dh l -> b (nh dh) l'), 'b d l -> b l d'
    )

  def forward(self, xm):
    x, m = xm
    q, k, v = self.wx(self.norm1(x)).split(self.d, -1)
    q, k, v = map(self.split, (self.rotate(q), self.rotate(k), v))
    A = F.softmax((einsum('bhri, bhrj -> bhij', q, k) + m) * self.c, -1)
    x = x + self.wo(self.concat(einsum('bhic, bhjc -> bhij', v, A)))
    return x + self.ff(self.norm2(x)), m


class GPT(nn.Module):
  def __init__(self, d, nh, nl, l, v):
    super().__init__()
    self.l = l 

    self.emb = nn.Embedding(v, d)
    self.out = nn.Linear(d, v, bias=False)

    self.rotate = RotaryEmbedding(l, d)

    self.layers = nn.Sequential(*[Layer(d, nh) for _ in range(nl)])
    for layer in self.layers:
      layer.rotate = self.rotate

    m = torch.tril(torch.ones(l, l)) - 1
    m[m == -1] = float('-inf')
    self.m = nn.Parameter(m, requires_grad=False)

    def fn(m):
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None: 
          nn.init.zeros_(m.bias)

    self.apply(fn)
    nn.init.normal_(self.emb.weight, 0, 0.02)

  def forward(self, T):
    assert len(T.shape) == 2, 'input must have 2 dimensions'
    _, l = T.shape
    assert l <= self.l, f'length must be <= {self.l}'
    xm = (self.emb(T), self.m[:l, :l])
    x, _ = self.layers(xm)
    return self.out(x)

  def loss(self, Tx, Ty):
    return F.cross_entropy(
      rearrange(self(Tx), 'b l v -> (b l) v'), rearrange(Ty, 'b l -> (b l)')
    )