import os
import sys
sys.path.append('..')
import ctypes

import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import graphnet as GNN
_cudart = ctypes.CDLL('libcudart.so')

NUM_ITERS = int(os.environ.get('NI')) #10
BATCH_SIZE = int(os.environ.get('BS')) #32 
sys.path.append(f'../utils')

D = int(os.environ.get('D')) #128
num_mp = int(os.environ.get('MP')) #15
mode = os.environ.get('MODE')
#M=1024


print("NI = ", NUM_ITERS, "BS = ", BATCH_SIZE, "D = ", D, "MP = ", num_mp, "mode = ", mode)
class PsuedoGraphNetBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp  = GNN.Mlp(2 * D, [D, D], layernorm=False) 
        self.edge_mlp  = GNN.Mlp(3 * D, [D, D], layernorm=False)
        #self.world_mlp = GNN.Mlp(3 * D, [D, D, D], layernorm=False)

    def forward(self, n, e):
        #n = self.node_mlp(torch.cat([n, n, n], dim=1))
        #e = self.edge_mlp(torch.cat([e, e, e], dim=1))
        #w = self.world_mlp(torch.cat([w, w, w], dim=1))
        n = self.node_mlp(n)
        e = self.edge_mlp(e)
        #w = self.world_mlp(w)
        return n, e

class PsuedoGraphNet(torch.nn.Module):
    def __init__(self, num_mp_steps=15):
        super().__init__()
        self.num_mp_steps = num_mp_steps
        self.blocks = torch.nn.ModuleList([
            PsuedoGraphNetBlock() for _ in range(num_mp_steps)])

    def forward(self, n, e):
        for block in self.blocks:
            n, e = block(n, e)
        return n, e


dev = torch.device('cuda:0')
net = PsuedoGraphNet(num_mp_steps=num_mp).half().to(dev)
#n = torch.randn(BATCH_SIZE * 1226,  D*3, device=dev, dtype=torch.float16)
#e = torch.randn(BATCH_SIZE * 12281, D*3, device=dev, dtype=torch.float16)
#w = torch.randn(BATCH_SIZE * 1660,  D*3, device=dev, dtype=torch.float16)
n = torch.randn(BATCH_SIZE * 40962,  D*2, device=dev, dtype=torch.float16)
e = torch.randn(BATCH_SIZE * 327660, D*3, device=dev, dtype=torch.float16)
#w = torch.randn(BATCH_SIZE * M,  D*3, device=dev, dtype=torch.float16)
if mode == 'prof':
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        profile_memory=True,
    ) as prof:
        t0 = time.perf_counter()
        for i in range(NUM_ITERS):
            net(n, e)
            prof.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
elif mode == 'ncu':
    with torch.no_grad():
        _cudart.cudaProfilerStart()
        t0 = time.perf_counter()
        for i in range(NUM_ITERS):
            net(n, e)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        _cudart.cudaProfilerStop()
else:
    #with torch.no_grad():
    t0 = time.perf_counter()
    for i in range(NUM_ITERS):
        net(n, e)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

print(f'Batch Size: {BATCH_SIZE}')
print(f'Num Iters: {NUM_ITERS}')
print(f'Elapsed time: {t1 - t0:.2f} seconds')
print(f'Throughput: {NUM_ITERS * BATCH_SIZE / (t1 - t0):.2f} samp/sec')

if mode == 'prof':
    print(prof \
         .key_averages(group_by_input_shape=False, group_by_stack_n=4) \
         .table(sort_by="self_cuda_time_total", row_limit=-1, top_level_events_only=False))
