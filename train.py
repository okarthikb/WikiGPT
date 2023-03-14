import torch, os, pickle, random, wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from gpt import *
from lion_pytorch import Lion
from torch.cuda.amp import GradScaler, autocast
from argparse import ArgumentParser


def process(gpu, node, gpus, world_size):
  rank = node * gpus + gpu
  
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
  print(f'started process {rank}\n')

  encoding = pickle.load(open('wikitext-103-tokens.pkl', 'rb'))
  split_size = len(encoding) // world_size
  encoding = encoding[:split_size * world_size]
  split = torch.tensor(encoding[rank * split_size:rank * split_size + split_size])
  
  d = 768
  nh = 12
  nl = 16
  l = 512
  v = 16384
  batch_size = 32
  steps = 30000
  lr = 1e-4

  torch.manual_seed(69)
  gpt = GPT(d, nh, nl, l, v).cuda(rank)
  optimizer = Lion(gpt.parameters(), lr)

  def generator():
    for _ in range(steps):
      indices = random.choices(range(len(split) - l - 1), k=batch_size)
      yield torch.stack([split[i:i + l + 1] for i in indices]).cuda(rank)

  if rank == 0:
    nparam = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
    print(f'{nparam} parameters\n')
    wandb.init(project='WikiGPT')
    wandb.run.name = 'rank 0'
    print()

  scaler = GradScaler()
  for step, batch in enumerate(generator(), 1):
    with autocast():
      loss = gpt.loss(batch[:, :-1], batch[:, 1:])
    
    if rank == 0:
      wandb.log({'loss': loss.item()})
      if step % 50 == 0:
        print(f'loss = {loss.item()}\tstep = {step}')
    
    scaler.scale(loss).backward()
    
    for p in gpt.parameters():
      if p.requires_grad:
        dist.reduce(tensor=p.grad.data, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
          p.grad.data /= world_size
        dist.broadcast(tensor=p.grad.data, src=0)
 
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

  if rank == 0:
    print()
    wandb.finish()
    torch.save(gpt.state_dict(), 'WikiGPT.pt')
 
  dist.destroy_process_group()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--nodes', type=int, default=1)
  parser.add_argument('--gpus', type=int)
  parser.add_argument('--node', type=int, default=0)
  args = parser.parse_args()

  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '6969'

  world_size = args.nodes * args.gpus
  
  mp.spawn(process, args=(args.node, args.gpus, world_size), nprocs=world_size)