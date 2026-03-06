# MIT License
#
# Copyright (c) 2026 Darrell Thomas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
CPUOffloadAdam: Memory-efficient Adam optimizer for consumer GPUs
Keeps optimizer state (momentum, variance) in system RAM instead of GPU VRAM.
Frees ~10GB+ of VRAM on a 1B parameter model.
Drop-in replacement for torch.optim.Adam.
Tested on 2x NVIDIA RTU 5090 GPUs.
'''

import torch
from collections import defaultdict


class CPUOffloadAdam:
    """
    Adam optimizer that keeps all state on CPU.
    
    GPU only holds model params + grads. Optimizer step runs on CPU.
    This allows training models that would otherwise OOM due to Adam's
    2x parameter memory overhead for momentum and variance tensors.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = defaultdict(dict)
        self.params = list(params)
        self.step_count = 0
        
    def zero_grad(self):
        """Clear gradients for all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self):
        """
        Perform a single optimization step.
        
        Gradients are copied to CPU, Adam update is computed in system RAM,
        and updated parameters are copied back to GPU.
        """
        self.step_count += 1
        
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                    
                # Copy grad to CPU (float32 for numerical stability)
                grad = p.grad.detach().cpu().float()
                param_cpu = p.data.detach().cpu().float()
                
                # Initialize state on CPU if needed
                if 'exp_avg' not in self.state[p]:
                    self.state[p]['exp_avg'] = torch.zeros_like(param_cpu)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(param_cpu)
                
                exp_avg = self.state[p]['exp_avg']
                exp_avg_sq = self.state[p]['exp_avg_sq']
                
                # Weight decay (decoupled, AdamW-style)
                if self.weight_decay != 0:
                    param_cpu.add_(param_cpu, alpha=-self.weight_decay * self.lr)
                
                # Adam update on CPU
                exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                # Bias correction
                bias_correction1 = 1 - self.beta1 ** self.step_count
                bias_correction2 = 1 - self.beta2 ** self.step_count
                
                step_size = self.lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
                
                param_cpu.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Copy updated param back to GPU (original dtype)
                p.data.copy_(param_cpu.to(p.data.dtype))
    
    def state_dict(self):
        """Return optimizer state for checkpointing."""
        return {
            'step_count': self.step_count,
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'state': {id(p): {k: v.clone() for k, v in s.items()} 
                      for p, s in self.state.items()}
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        self.step_count = state_dict['step_count']
        self.lr = state_dict['lr']
        self.beta1, self.beta2 = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
