import logging
import random
from typing import Any, Optional

import torch.utils.data
from ytreader import YTTableParallelReader

logger = logging.getLogger(__name__)


class YTDataset(torch.utils.data.IterableDataset):
    """Reads dataset in chunks from internal yandex database. Assumes that data is already shuffled."""

    def __init__(
        self,
        cluster: str,
        table: str,
        chunk_size: int = 10 ** 4,
        cache_size: int = 10 ** 3,
        num_readers: int = 4,
        max_consecutive_errors: int = 10,
        seed: Optional[Any] = None,
        cycle: bool = True,
    ):
        self.cluster, self.table = cluster, table
        self.chunk_size, self.num_readers, self.cache_size = chunk_size, num_readers, cache_size
        self.random_state, self.cycle = random.Random(seed), cycle
        self.max_consecutive_errors = max_consecutive_errors

    def shuffle_data_sources(self, seed: int):
        return YTDataset(
            self.cluster,
            self.table,
            self.chunk_size,
            self.num_readers,
            self.cache_size,
            seed=seed,
            cycle=self.cycle,
        )

    def __iter__(self):
        started = False
        consecutive_errors = 0
        logger.info("Pre-fetching training samples...")

        while True:
            try:
                reader = YTTableParallelReader(self.cluster, self.table, self.cache_size, self.num_readers)
                reader.reset_to_row(0)
                start = self.random_state.randint(0, reader.num_rows - self.chunk_size - 1)
                for i, row in enumerate(reader.make_subset_reader(start, start + self.chunk_size)):
                    if not started:
                        logger.info("Began iterating minibatches!")
                        started = True
                    yield start + i, row
                    consecutive_errors = 0
                if not self.cycle:
                    raise StopIteration()
            except StopIteration:
                logger.exception(f"YTDataset finished iteration!")
                raise
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    raise e
                else:
                    logger.exception(f"Caught {e}, retrying from a different chunk.")
                    continue
