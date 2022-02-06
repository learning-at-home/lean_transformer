import ctypes
import logging
import multiprocessing as mp
import multiprocessing.sharedctypes
from collections import defaultdict
from functools import partial
from typing import Optional, Union

import torch.utils.data
from datasets import IterableDataset, disable_progress_bar
from transformers import GPT2TokenizerFast

from .yt_streaming import YTDataset

logger = logging.getLogger(__name__)


disable_progress_bar()


def split_list(l, n):
    # splits list/string into n size chunks
    return (l[i : i + n] for i in range(0, len(l), n))


def process_instance(tokenizer, text, max_seq_length):
    tokenized_text = tokenizer.encode(text) + [tokenizer.eos_token_id]

    for chunk in split_list(tokenized_text, max_seq_length):
        yield chunk


def examples_from_documents(tokenizer, documents, max_sequence_length: mp.sharedctypes.Synchronized):
    texts = (text for text in documents["text"] if len(text) > 0 and not text.isspace())

    new_examples = defaultdict(list)

    for text in texts:
        try:
            instances = process_instance(tokenizer, text, int(max_sequence_length.value))

            for instance in instances:
                new_examples["input_ids"].append(instance)
        except Exception as e:
            logger.warning(f"Caught {repr(e)}, ignoring...", exc_info=True)

    return new_examples


def make_training_dataset(
    tokenizer: GPT2TokenizerFast,
    max_sequence_length: Union[int, mp.sharedctypes.Synchronized],
    shuffle_buffer_size: int = 10 ** 4,
    shuffle_seed: Optional[int] = None,
    preprocessing_batch_size: int = 256,
):
    if not isinstance(max_sequence_length, mp.sharedctypes.Synchronized):
        assert isinstance(max_sequence_length, int)
        max_sequence_length = mp.Value(ctypes.c_int64, max_sequence_length)
    assert isinstance(tokenizer, GPT2TokenizerFast)
    dataset = YTDataset("hahn", "//home/gena/datasets/pile/train")
    text_col = b"Text"

    def extract_training_columns(batch):
        return dict(text=[bytes.decode(row, errors="ignore") for row in batch[text_col]])

    dataset = IterableDataset(dataset).map(extract_training_columns, batched=True, batch_size=preprocessing_batch_size)

    dataset = dataset.map(
        partial(examples_from_documents, tokenizer, max_sequence_length=max_sequence_length),
        batched=True,
        batch_size=preprocessing_batch_size,
    )

    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    dataset = dataset.with_format("torch")
    return WrappedIterableDataset(dataset)


class WrappedIterableDataset(torch.utils.data.IterableDataset):
    """Wraps huggingface IterableDataset as pytorch IterableDataset, implement default methods for DataLoader"""

    def __init__(self, hf_iterable, verbose: bool = True):
        self.hf_iterable = hf_iterable
        self.verbose = verbose

    def __iter__(self):
        started = False
        logger.info("Pre-fetching training samples...")
        while True:
            for sample in self.hf_iterable:
                if not started:
                    logger.info("Began iterating minibatches!")
                    started = True
                yield sample
