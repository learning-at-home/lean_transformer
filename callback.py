import os.path
from typing import Any

import hivemind
import torch
import transformers
from transformers import TrainingArguments

from arguments import TrainingPeerArguments
from tasks.mlm.task import MLMTrainingTask
from utils import LocalMetrics, logger


class CollaborativeCallback(transformers.TrainerCallback):
    """
    This callback monitors and reports collaborative training progress,
    In case of a catastrophic failure, it can also revert training to a backup
    """

    def __init__(self, task: MLMTrainingTask, args: TrainingPeerArguments):
        super().__init__()
        self.task = task
        self.dht, self.collaborative_optimizer = task.dht, task.collaborative_optimizer
        self.statistics_expiration = args.statistics_expiration
        self.last_reported_epoch = -1
        self.samples = 0
        self.mini_steps = 0  # number of calls to optimizer.step, NOT equal to the number of global steps
        self.loss = 0
        self.total_samples_processed = 0
        self.backup_every_epochs = args.backup_every_epochs
        self.state_path = args.state_path

    def on_train_begin(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        if os.path.isfile(self.state_path):
            self.restore_from_backup(self.state_path)
            logger.info("Loaded state")

        logger.info("Loading state from peers")
        self.collaborative_optimizer.load_state_from_peers()

        if os.path.isfile(self.state_path):
            self.restore_from_backup(self.state_path, check_epoch=True)

    def on_step_end(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        self.task.update_sequence_length()

        control.should_log = True
        if not self.params_are_finite():
            if not os.path.exists(self.state_path):
                raise RuntimeError("Encountered broken parameters, but there is no backup to fall back to.")
            logger.warning("Parameters are invalid, reloading model from earlier state")
            self.restore_from_backup(self.state_path)
            return control

        if state.log_history:
            self.loss += state.log_history[-1]["loss"]
            self.mini_steps += 1
            if self.collaborative_optimizer.local_epoch != self.last_reported_epoch:
                self.last_reported_epoch = self.collaborative_optimizer.local_epoch
                self.total_samples_processed += self.samples
                samples_per_second = self.collaborative_optimizer.tracker.performance_ema.samples_per_second
                statistics = LocalMetrics(
                    epoch=self.collaborative_optimizer.local_epoch,
                    samples_per_second=samples_per_second,
                    samples_accumulated=self.samples,
                    loss=self.loss,
                    mini_steps=self.mini_steps,
                )
                logger.info(f"Current epoch: {self.collaborative_optimizer.local_epoch}")
                logger.info(f"Your current contribution: {self.total_samples_processed} samples")
                logger.info(f"Performance: {samples_per_second} samples/sec")
                if self.mini_steps:
                    logger.info(f"Local loss: {self.loss / self.mini_steps}")

                self.loss = 0
                self.mini_steps = 0
                if self.collaborative_optimizer.local_epoch == self.collaborative_optimizer.tracker.global_epoch:
                    self.dht.store(
                        key=self.collaborative_optimizer.run_id + "_metrics",
                        subkey=self.task.local_public_key,
                        value=statistics.dict(),
                        expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                        return_future=True,
                    )
                if (
                    self.backup_every_epochs is not None
                    and self.collaborative_optimizer.local_epoch % self.backup_every_epochs == 0
                ):
                    self.backup_state()

        self.samples = self.collaborative_optimizer.grad_averager.local_samples_accumulated

        return control

    @torch.no_grad()
    def params_are_finite(self):
        for param in self.task.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True

    @torch.no_grad()
    def backup_state(self) -> Any:
        logger.info("Saving backup")
        return torch.save(
            {
                "model": self.task.model.state_dict(),
                "training": self.collaborative_optimizer.state_dict(),
                "scheduler": self.collaborative_optimizer.state_averager.scheduler.state_dict(),
                "local_epoch": self.collaborative_optimizer.local_epoch,
            },
            self.state_path,
        )

    @torch.no_grad()
    def restore_from_backup(self, path, check_epoch=False):
        state = torch.load(path)
        current_epoch = self.collaborative_optimizer.local_epoch
        backup_epoch = state["local_epoch"]
        if not check_epoch or backup_epoch >= current_epoch:
            self.task.model.load_state_dict(state["model"], strict=False)
            self.collaborative_optimizer.load_state_dict(state["training"])
            self.collaborative_optimizer.state_averager.scheduler.load_state_dict(state["scheduler"])
            self.collaborative_optimizer.state_averager.local_epoch = backup_epoch
            logger.info("Restored from a backup")
        else:
            logger.info("Bypassed restoring state from local backup: backup state is too old.")
