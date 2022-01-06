import torch
import transformers
from torch.distributed.distributed_c10d import _get_default_group
from transformers import TrainerControl, TrainerState, TrainingArguments

import arguments
import tasks

AUTHORITATIVE_RANK = 0
BROADCAST_BUFFER_SIZE: int = 250 * 1024 * 1024


def is_main_process() -> bool:
    """Whether this is the main process on **this peer's** distributed run. Non-distributed process is always main."""
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == AUTHORITATIVE_RANK


class SynchronizationCallback(transformers.TrainerCallback):
    """Minimalistic callback for non-master DDP workers"""

    def __init__(self, task: "tasks.TrainingTaskBase", args: "arguments.TrainingPeerArguments"):
        self.task = task
        self.is_master = is_main_process()
        self._checksum_counter = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.distributed.is_initialized():
            self._sync_params_and_buffers()

    def on_step_end(
            self,
            args: TrainingArguments,
            state: transformers.TrainerState,
            control: transformers.TrainerControl,
            **kwargs,
    ):
        control.should_log = True
        model = self.task.model
        if torch.distributed.is_initialized():
            self._sync_params_and_buffers()

            self._checksum_counter += 1
            if self._checksum_counter % 100 == 0:
                rank = torch.distributed.get_rank()
                print(end=f"CHECKSUM({rank})={float(sum(p.sum().item() for p in model.parameters()))}\n")
        self.task.on_step_end()

    def _sync_params_and_buffers(self):
        """Synchronize model params and buffers from master"""
        #TODO -- check and run this ONLY if params have changed
        module_states = []
        for name, param in self.task.model.state_dict().items():
            module_states.append(param)

        if module_states:
            torch.distributed._broadcast_coalesced(
                _get_default_group(), module_states, BROADCAST_BUFFER_SIZE, AUTHORITATIVE_RANK
            )
