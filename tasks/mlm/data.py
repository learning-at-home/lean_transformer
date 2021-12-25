import random
from collections import defaultdict
from functools import partial
from typing import Optional

import torch.utils.data
from datasets import IterableDataset, interleave_datasets
from hivemind.utils.logging import get_logger
from razdel import sentenize

from tasks.mlm.yt_streaming import YTDataset

logger = get_logger(__name__)


def make_training_dataset(
    tokenizer,
    shuffle_buffer_size: int = 10 ** 4,
    shuffle_seed: Optional[int] = None,
    preprocessing_batch_size: int = 256,
    max_sequence_length: int = 512,
):
    runet = YTDataset("hahn", "//home/yr/nlp/big_russian_bert/common_mincount1_nolimit_nodedup")
    wiki = YTDataset("hahn", "//home/yr/nlp/big_russian_bert/wikipedia")
    taiga = YTDataset("hahn", "//home/yr/nlp/big_russian_bert/taiga")
    librusec = YTDataset("hahn", "//home/yr/nlp/big_russian_bert/rdt")

    datasets = dict(runet=runet, wiki=wiki, taiga=taiga, librusec=librusec)
    weights = dict(runet=0.7, wiki=0.1, taiga=0.1, librusec=0.1)
    colnames = dict(runet=b"data", wiki=b"Text", taiga=b"text", librusec=b"text")

    def extract_training_columns(key, batch):
        texts = [bytes.decode(row, errors="ignore") for row in batch[colnames[key]]]
        return dict(text=texts, key=[key] * len(texts))

    datasets = {key: IterableDataset(dataset) for key, dataset in datasets.items()}
    datasets = {
        key: dataset.map(
            partial(extract_training_columns, key),
            batched=True,
            batch_size=preprocessing_batch_size,
        )
        for key, dataset in datasets.items()
    }
    datasets = {
        key: dataset.map(
            partial(tokenize_function, tokenizer, max_sequence_length=max_sequence_length),
            batched=True,
            batch_size=preprocessing_batch_size,
        )
        for key, dataset in datasets.items()
    }

    dataset = interleave_datasets(
        [datasets[k] for k in sorted(datasets.keys())],
        probabilities=[weights[k] for k in sorted(datasets.keys())],
    )

    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    dataset = dataset.with_format("torch")
    return WrappedIterableDataset(dataset)


def create_instances_from_document(tokenizer, document, max_sequence_length):
    """Creates `TrainingInstance`s for a single document."""
    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0

    segmented_sents = [s.text for s in sentenize(document)]

    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))
        if i == len(segmented_sents) - 1 or current_length >= max_sequence_length:
            if len(current_chunk) > 1:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.append(current_chunk[j])

                tokens_b = []

                for j in range(a_end, len(current_chunk)):
                    tokens_b.append(current_chunk[j])

                if random.random() < 0.5:
                    # Random next
                    is_random_next = True
                    # in this case, we just swap tokens_a and tokens_b
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    # Actual next
                    is_random_next = False

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                instance = tokenizer(
                    " ".join(tokens_a),
                    " ".join(tokens_b),
                    padding="max_length",
                    truncation="longest_first",
                    max_length=max_sequence_length,
                    # We use this option because DataCollatorForLanguageModeling
                    # is more efficient when it receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
                assert len(instance["input_ids"]) <= max_sequence_length
                instance["sentence_order_label"] = 1 if is_random_next else 0
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances


def tokenize_function(tokenizer, examples, max_sequence_length):
    # Remove empty texts
    texts = [text for text in examples["text"] if len(text) > 0 and not text.isspace()]
    new_examples = defaultdict(list)

    for text in texts:
        instances = create_instances_from_document(tokenizer, text, max_sequence_length)
        for instance in instances:
            for key, value in instance.items():
                new_examples[key].append(value)

    return new_examples


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
