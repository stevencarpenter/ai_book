# Memory Optimization for Loss Landscape Visualization

## The Problem

Running loss landscape notebooks (like `2_wormhole.ipynb`) on consumer GPUs (16-32GB VRAM) may cause **CUDA Out of Memory** errors during `optimizer.step()`.

This happens because Adam stores 2x the model parameters in VRAM for momentum and variance buffers. On a 1B parameter model, that's an extra ~8-10GB of VRAM on top of the model weights, activations, and gradients.

## Before You Start

**Critical:** GPU memory persists even after closing notebook tabs. The kernel keeps running in the background.

1. In JupyterLab, go to the left sidebar → **"Running Terminals and Kernels"** (the square stop icon)
2. Click **"Shut Down All"** to kill all notebook kernels
3. Open **only** the notebook you want to run
4. **Kernel → Restart & Clear All Outputs**

Verify with `nvidia-smi` in a terminal — GPU memory should be nearly empty before starting.

## Quick Fix

### Step 1: Add a new cell at the very top of the notebook

Insert this as the **first cell** and run it before anything else:

```python
import torch
import gc
import sys
import os

# Import the memory-optimized optimizer
sys.path.insert(0, 'memory_optimize')
from cpu_offload_adam import CPUOffloadAdam

# Help PyTorch manage memory better
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clear_gpu():
    """Clear GPU memory from current process."""
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    print(f"GPU memory: {torch.cuda.memory_allocated(0)/1e9:.1f}GB used")

clear_gpu()
```

You should see `GPU memory: 0.0GB used` (or very close to it).

### Step 2: Replace the optimizer

Find the line that creates the Adam optimizer (usually looks like this):

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

Replace it with:

```python
optimizer = CPUOffloadAdam(model.parameters(), lr=lr)
```

**That's it.** Run the notebook normally.

## How It Works

`CPUOffloadAdam` is a drop-in replacement for `torch.optim.Adam` that:

1. Keeps optimizer state (momentum and variance buffers) in **system RAM** instead of GPU VRAM
2. Copies gradients to CPU, performs the Adam update on CPU
3. Copies updated parameters back to GPU

This frees ~10GB+ of VRAM on a 1B parameter model.

## Performance

| Operation | torch.optim.Adam | CPUOffloadAdam |
|-----------|------------------|----------------|
| optimizer.step() | ~5ms | ~60-120ms |
| Forward pass | ~540ms | ~540ms |

The optimizer step is slower due to CPU-GPU data transfer, but **loss landscape visualization is dominated by forward passes** (1024 per training step for a 32x32 grid). Total runtime impact is negligible.

## Tested On

- NVIDIA RTX 5090 (32GB VRAM)
- PyTorch 2.10 + CUDA 13
- Llama 3.2 1B
- Ubuntu 24.04, 512GB system RAM

Should work on any GPU with 16GB+ VRAM and sufficient system RAM.

## Troubleshooting

**Still getting OOM?**
- Make sure you shut down ALL other Jupyter kernels first
- Run `nvidia-smi` to check what's using GPU memory
- Kill any zombie Python processes: `pkill -f python` (careful: this kills all Python)

**Import error for CPUOffloadAdam?**
- Make sure the `memory_optimize` folder is in the same directory as the notebook
- Check the `sys.path.insert` line points to the correct path

## Author

Darrell Thomas — [github.com/darrellthomas](https://github.com/darrellthomas)

Built while working through Stephen Welch's excellent [Illustrated Guide to AI](https://github.com/stephencwelch/ai_book).

Developed with assistance from Opus 4.5 (Anthropic).
