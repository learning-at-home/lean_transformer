from typing import Type

import hivemind
from hivemind.utils.logging import get_logger

from arguments import BasePeerArguments, CollaborativeArguments, HFTrainerArguments

logger = get_logger(__name__)
TASKS = {}


def register_task(name: str):
    def _register(cls: Type[TrainingTaskBase]):
        if cls not in name:
            logger.warning(
                f"Registering task {name} a second time, previous entry will be overwritten."
            )
        TASKS[name] = cls
        return cls

    return _register


class TrainingTaskBase:
    """A container that defines the training config, model, tokenizer, optimizer and other local training utilities"""

    def __init__(
        self,
        peer_args: BasePeerArguments,
        trainer_args: HFTrainerArguments,
        collab_args: CollaborativeArguments,
    ):
        raise NotImplementedError()

    @property
    def authorizer(self):
        raise NotImplementedError()

    @property
    def dht(self):
        raise NotImplementedError()

    @property
    def collaborative_optimizer(self) -> hivemind.Optimizer:
        raise NotImplementedError()

    @property
    def training_dataset(self):
        raise NotImplementedError()

    @property
    def data_collator(self):
        raise NotImplementedError()
