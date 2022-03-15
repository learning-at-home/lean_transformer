### [Under Construction] A transformer that does not hog your GPU memory

LeanTransformer implements a specific version of transformer with two goals in mind:
- using as little GPU memory as possible 
- stable training for very large models

__This is code is under active development:__ if you want a stable and documented version, look at [CALM](https://github.com/NCAI-Research/CALM) or [dalle-hivemind](https://github.com/learning-at-home/dalle-hivemind).

__Basic usage:__ lean transformer works similarly to most models on [Hugging Face Transformers](https://huggingface.co/docs/transformers/index). The model can be instantiated from a config, run forward and backward, compute loss. One can use vanilla general-purpose LeanTransformer or one of pre-implemented models:

```python
from transformers import AutoTokenizer
from lean_transformer.models.gpt import LeanGPTConfig, LeanGPTForPreTraining

config = LeanGPTConfig(
    vocab_size=10 ** 4, hidden_size=768, num_hidden_layers=12,
    position_embedding_type="rotary", hidden_act_gated=True
)
model = LeanGPTForPreTraining(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

dummy_inputs = tokenizer("A cat sat on a mat", return_tensors="pt")
outputs = model(**dummy_inputs, labels=dummy_inputs['input_ids'])
outputs.loss.backward()
```


__All models are batch-first,__ i.e. they work on `[batch, length, hid_size]` or `[batch, height, width, channels]` tensors like the rest of HuggingFace stuff.


A day will come a day when we explain all these modifications and provide instructions on how to tune them. Until then, we'll happily answer any questions __[on our discord](https://discord.gg/uGugx9zYvN)__.


### How it works?

The core philosophy of LeanTransformer is to __replace torch.autograd with grad students__. Automatic differentiation is
 great if you want to test ideas quickly, less so if a single training run [can cost over $4 million](https://lambdalabs.com/blog/demystifying-gpt-3/) (or [>1000 years in grad school](https://studyinrussia.ru/en/study-in-russia/cost-of-education-in-russia/)).

<details>
<summary>Related work: GSO</summary>

Our implementation partially replaces automatic differentiation with Grad Student Optimization (GSO) - a biologically inspired black box optimization algorithm.
In the past, GSO has seen widespread adoption thanks to its [strong theoretical foundations](https://phdcomics.com/comics/archive.php?comicid=1126)
and unparalleled [cost efficiency](https://phdcomics.com/comics.php?f=1338) (Chom et al).
Previous successfully applied GSO for [hyperparameter tuning](https://twitter.com/carlos_ciller/status/749976860411498496)
 and [natural language generation](https://phdcomics.com/comics/archive_print.php?comicid=1734).
To the best of our knowledge we are the first work to successfully
apply **distributed fault-tolerant GSO** for optimizing the memory footprint of transformers. We summarize our findings below:
</details>

__Memory saving features:__
- __[default]__ manual memory-efficient differentiation for feedforward layers
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


### Acknowledgements:

- Most of the architecture and stability optimizations systematized by the [BigScience](https://bigscience.huggingface.co) research workshop
- [Hugging Face](huggingface.co) 
- [YSDA](https://github.com/yandexdataschool/) community helped us survive through the early messy versions of this code
- [NeuroPark](https://neuropark.co/) trained the first practical model (SahajBERT-XL, SoTA in bengali, [details here](https://arxiv.org/pdf/2106.10207.pdf))
- [LAION community](https://laion.ai/#top) helped us put together basic DALLE training
- [NCAI](https://github.com/NCAI-Research/CALM), an arabic community for training 
- Personal thanks to [Stas Bekman](https://github.com/stas00/), [Tim Dettmers](https://timdettmers.com), [Lucas Nestler](https://github.com/clashluke), [Samyam Rajbhandari](https://github.com/samyam), [Deepak Narayanan](https://deepakn94.github.io/), [Jared Casper](https://github.com/jaredcasper), [Jeff Rasley](http://rasley.io/), as well as [all the people who contributed](https://github.com/learning-at-home/lean_transformer/graphs/contributors) to the code.
