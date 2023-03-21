import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from torch.optim.lr_scheduler import _LRScheduler


class RMSNorm(nn.Module):
  def __init__(self, d, eps=1e-8):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(d))

  def forward(self, x):
    rrms = torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) 
    return self.weight * x * rrms


class RotaryEmbedding(nn.Module):
  def __init__(self, l, d):
    super().__init__()
    θ = 1e4 ** (-repeat(torch.arange(0, d, 2), 'i -> (i j)', j=2) / d)
    lθ = einsum('i, j -> ij', torch.arange(l), θ)
    self.sin = nn.Parameter(lθ.sin(), requires_grad=False)
    self.cos = nn.Parameter(lθ.cos(), requires_grad=False)

  def forward(self, x):
    _, l, _ = x.shape
    y2 = torch.cat((-x[..., 1::2, None], x[..., ::2, None]), -1)
    y = rearrange(y2, '... x y -> ... (x y)')
    return x * self.cos[:l] + y * self.sin[:l]


class Layer(nn.Module):
  def __init__(self, d, nh, pe=lambda x: x):
    super().__init__()
    self.scale = d ** -0.5
    self.wx, self.wo = nn.Linear(d, 3 * d), nn.Linear(d, d) 
    self.mhanorm, self.ffnorm = RMSNorm(d), RMSNorm(d)
    self.pe = pe
    self.split = lambda x: rearrange(x, 'b l (nh dh) -> b nh l dh', nh=nh)

    h = 8 * d // 3
    self.u, self.v = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False)
    self.w = nn.Linear(h, d, bias=False)
    self.ff = lambda x: self.w(F.silu(self.u(x)) * self.v(x))

  def forward(self, xm):
    x, m = xm
    q, k, v = rearrange(self.wx(self.mhanorm(x)), 'b l (n d) -> n b l d', n=3)
    q, k, v = map(self.split, (self.pe(q), self.pe(k), v))
    A = F.softmax(einsum('bhic, bhjc -> bhij', q, k) * self.scale + m, -1)
    x = x + self.wo(rearrange(A @ v, 'b nh l dh -> b l (nh dh)'))
    return x + self.ff(self.ffnorm(x)), m


class GPT(nn.Module):
  def __init__(self, d, nh, nl, l, v):
    super().__init__()
    self.l = l 

    self.emb = nn.Embedding(v, d)
    self.out = nn.Linear(d, v, bias=False)

    self.rope = RotaryEmbedding(l, d)
    self.layers = nn.Sequential(*[Layer(d, nh, self.rope) for _ in range(nl)])

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

    self.np = sum(p.numel() for p in self.parameters() if p.requires_grad)

  def param_groups(self, weight_decay=1e-2):
    decay, nodecay = set(), set()

    for mn, m in self.named_modules():
      for pn, p in m.named_parameters():
        fpn = f'{mn}.{pn}' if mn else pn
        if pn.endswith('bias'):
          nodecay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, nn.Linear):
          decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, (nn.Embedding, RMSNorm)):
          nodecay.add(fpn)
   
    pdict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    assert len(decay & nodecay) == 0, f'{decay & nodecay}'
    assert len(pdict.keys() - (decay | nodecay)) == 0
    decay = [pdict[pn] for pn in sorted(list(decay))]
    nodecay = [pdict[pn] for pn in sorted(list(nodecay))]

    return [
      {'params': decay, 'weight_decay': weight_decay},
      {'params': nodecay, 'weight_decay': 0}
    ]

  def forward(self, x):
    _, l = x.shape 
    x, _ = self.layers((self.emb(x), self.m[:l, :l]))
    return self.out(x)

  def loss(self, x, y):
    return F.cross_entropy(
      rearrange(self(x), 'b l v -> (b l) v'), rearrange(y, 'b l -> (b l)')
    )


class InvSqrtLR(_LRScheduler):
  def __init__(self, optimizer, max_lr, min_lr, warmup, steps):
    d = steps ** 0.5 - warmup ** 0.5
    scale = (max_lr - min_lr) * (steps * warmup) ** 0.5 / d
    shift = (min_lr * steps ** 0.5 - max_lr * warmup ** 0.5) / d

    self.cur_step = 0
    self.optimizer, self.steps = optimizer, steps
    self.schedule = lambda x: min(
      lambda x: max_lr * x / warmup, lambda x: scale * (1 / x) ** 0.5 + shift
    )

    super().__init__(optimizer) 
 
  def get_lr(self):
    return [group['lr'] for group in self.optimizer.param_groups]
  
  def step(self):
    assert self.cur_step <= self.steps, 'cannot exceed max steps'
    self.cur_step += 1
    lr = self.schedule(self.cur_step)
    for group in self.optimizer.param_groups:
      group['lr'] = lr
    super().step(None)
