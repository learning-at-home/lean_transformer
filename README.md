### A transformer that does not hog your GPU memory

__This is a preview version:__ if you want a stable and documented version, look at [CALM](https://github.com/NCAI-Research/CALM) instead.

LeanTransformer implements a specific version of transformer with two goals in mind:
- using as little GPU memory as possible 
- not blowing up for GPT-3-sized models

TL;DR memory saving features:
- __[default]__ manually optimized autograd for FFN and attention layers
- __[option]__ gradient checkpointing [(Griewank et al, ](https://dl.acm.org/doi/10.1145/347837.347846) [Chen et al, 2016)](https://arxiv.org/pdf/1604.06174.pdf)
- __[option]__ reversible layers using ClashLuke's [revlib](https://github.com/clashluke/revlib), based on [(Gomez et al, 2017, ](https://proceedings.neurips.cc/paper/2017/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf) [Kitaev et al, 2020)](https://arxiv.org/abs/2001.04451)
- __[option]__ customizable parameter sharing [(Radford et al, 2019,](https://arxiv.org/abs/1909.11942) [Xue et al, 2021)](https://arxiv.org/abs/2107.11817)
- __[option]__ CPU-offloaded 8-bit LAMB [(Dettmers et al, 2021)](https://arxiv.org/abs/2110.02861) 
- A pinch of magic that we'll explain eventually [(hopefully)](https://quotefancy.com/quote/39802/Mikhail-Bulgakov-Yes-man-is-mortal-but-that-would-be-only-half-the-trouble-The-worst-of)

Testing for correctness:
- ```PYTHONPATH=.. pytest .```

Not implemented:
- In reversible mode, one can further save memory by computing backward in chunks:
  - a few tokens at a time for feedforward layers, since `grad(concat(mlp(x1), mlp(x2))) = concat(grad(mlp(x1)), grad(mlp(x2)))`
  - a few heads at a time for self-attention, since `grad(head1 + head2) = grad(head1) + grad(head2)`, where head1 and head2 are attention outputs *after linear projection*
- Attention could be computed in `O(sqrt(n))` memory [(Rabe et al, 2021)](https://arxiv.org/abs/2112.05682)
- No sparse or linear attention: for large models, attention is not a bottleneck for typical NLP tasks (up to [length 2048](https://arxiv.org/abs/2005.14165))

A day will come a day when we explain all these modifications and provide instructions on how to tune them.
[But it is not this day!](https://youtu.be/3Ri0-fahanU?t=25). If you want 

