### A transformer that does not hog your GPU memory

__This is a preview version:__ if you want a stable and documented version, look at [CALM](https://github.com/NCAI-Research/CALM) instead.

LeanTransformer implements a specific version of transformer with two goals in mind:
- using as little GPU memory as possible 
- stable training for GPT-3-sized models

Testing for correctness:
- ```PYTHONPATH=. pytest ./tests```

<details>
<summary>Readme under construction</summary>

The core philosophy of LeanTransformer is to __use grad students instead of torch.autograd__. Automatic differentiation is
 great if you want to test ideas quickly, but less so if a single training run can cost [millions](https://lambdalabs.com/blog/demystifying-gpt-3/).

<details>
<summary>Related work: GSO</summary>

Our implementation partially replaces automatic differentiation with Grad Student Optimization (GSO) - a biologically inspired black box optimization algorithm.
Prior work (Chom et al) successfully adopted GSO for [hyperparameter tuning](https://twitter.com/carlos_ciller/status/749976860411498496)
 and [ill-posed problems](https://encyclopediaofmath.org/wiki/Ill-posed_problems).
GSO has a [strong theoretical foundation](https://phdcomics.com/comics/archive.php?comicid=1126)
and unparalleled [cost efficiency](https://phdcomics.com/comics.php?f=1338).
To the best of our knowledge we are the first work to successfully
apply **distributed fault-tolerant GSO** for optimizing the memory footprint of transformers. We summarize our findings below:
</details>

__Memory saving features:__
- __[default]__ manual memory-efficient differentiation for FFN and attention layers
- __[option]__ gradient checkpointing [(Griewank et al, ](https://dl.acm.org/doi/10.1145/347837.347846) [Chen et al, 2016)](https://arxiv.org/pdf/1604.06174.pdf)
- __[option]__ reversible layers using ClashLuke's [revlib](https://github.com/clashluke/revlib), based on [(Gomez et al, 2017, ](https://proceedings.neurips.cc/paper/2017/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf) [Kitaev et al, 2020)](https://arxiv.org/abs/2001.04451)
- __[option]__ PixelFly block-sparse layers that significantly reduce the number of parameters [(Chen et al, 2021)](https://arxiv.org/abs/2112.00029)
- __[option]__ customizable parameter sharing [(Radford et al, 2019,](https://arxiv.org/abs/1909.11942) [Xue et al, 2021)](https://arxiv.org/abs/2107.11817)
- __[option]__ CPU-offloaded 8-bit LAMB [(Dettmers et al, 2021)](https://arxiv.org/abs/2110.02861) 
- A pinch of magic that we'll explain eventually [(hopefully)](https://quotefancy.com/quote/39802/Mikhail-Bulgakov-Yes-man-is-mortal-but-that-would-be-only-half-the-trouble-The-worst-of)

__Other features:__
- __[default]__ Pre-normalization: a more stable layer order used in [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (as opposed to the [original transformer](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html))
- __[option]__ Sandwich Norm, as proposed in [(Ding et al, 2021)](https://arxiv.org/pdf/2105.13290.pdf)
- __[option]__ Maintaining FP32 residuals in mixed precision training, learned from discussions with [Samyam](https://www.microsoft.com/en-us/research/people/samyamr/) and [Jeff](https://www.microsoft.com/en-us/research/people/jerasley/) from [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- __[option]__ Rotary Position Embeddings, proposed by [Su et al](https://arxiv.org/abs/2104.09864) and [popularized by EleutherAI](https://blog.eleuther.ai/rotary-embeddings/)
- __[option]__ Gated activations (e.g. GeGLU) [(Shazeer et al, 2020)](https://arxiv.org/abs/2002.05202), based on [(Dauphin et al, 2016)](https://arxiv.org/abs/1612.08083)
- __[option]__ Sequence length warmup aka Curriculum Learning [(Li et al, 2021)](https://arxiv.org/abs/2108.06084)

__Not implemented:__
- In reversible mode, one can further save memory by computing backward in chunks:
  - a few tokens at a time for feedforward layers, since `grad(concat(mlp(x1), mlp(x2))) = concat(grad(mlp(x1)), grad(mlp(x2)))`
  - a few heads at a time for self-attention, since `grad(head1 + head2) = grad(head1) + grad(head2)`, where head1 and head2 are attention outputs *after linear projection*
- Attention could be computed in `O(sqrt(n))` memory [(Rabe et al, 2021)](https://arxiv.org/abs/2112.05682)
- No sparse or linear attention: they are great for very long sequences. However, for large models, **attention is not a bottleneck** in typical NLP and vision tasks (tested gpt-3 up to length 4096).
- Per-block grad scaling as described in [(Ramesh et al, 2021)](https://arxiv.org/pdf/2102.12092.pdf) - we rely on Sandwich Norm to maintain stability up to 96 layers (did not test more). However, it would be nice to 
  have per-block scaling to avoid the need for an extra LayerNorm.
- Something else that we missed - please find us [on discord](https://discord.gg/uGugx9zYvN).

A day will come a day when we explain all these modifications and provide instructions on how to tune them.
[But it is not this day!](https://youtu.be/3Ri0-fahanU?t=25). Until then, we'll happily answer any questions __[on our discord](https://discord.gg/uGugx9zYvN)__.

### Running the code
__[under constructuion]__ - use the instructions from CALM readme

### Acknowledgements:

- Most of the architecture and stability optimizations were learned through the [BigScience](https://bigscience.huggingface.co) research workshop
- [YSDA](https://github.com/yandexdataschool/) community helped us survive through the early messy versions of this code
- [NeuroPark](https://neuropark.co/) trained the first practical model (SahajBERT-XL, SoTA in bengali, [details here](https://arxiv.org/pdf/2106.10207.pdf))
- TODO DALLE community: at least mention the demo, maybe we end up training something even cooler
- TODO NCAI community: ask them how best to acknowledge them
- TODO Hugging Face: ask them how best to acknowledge them
- TODO Personal: stas00, samyam, jared, more? (this does not include co-authors: Tim,Lucile,Quentin,Denis,Gennady,etc; also, this does not include hivemind contributors)

</details>