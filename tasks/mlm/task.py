import ctypes
import os
from dataclasses import asdict
from pathlib import Path

import hivemind
import torch.optim
import transformers
from hivemind import Float16Compression, SizeAdaptiveCompression, Uniform8BitQuantization
from hivemind.optim.experimental.state_averager import LRSchedulerBase, ParamGroups
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

import utils
from arguments import BasePeerArguments, CollaborativeArguments, HFTrainerArguments
from huggingface_auth import authorize_with_huggingface
from lib.models import LeanAlbertConfig, LeanAlbertForPreTraining
from lib.training.lamb_8bit import CPULAMB8Bit
import multiprocessing as mp

from .whole_word_mask import DataCollatorForWholeWordMask
from .data import make_training_dataset

hivemind.use_hivemind_log_handler("in_root_logger")
logger = hivemind.get_logger()


class MLMTrainingTask:
    """A container that defines the training config, model, tokenizer, optimizer and other local training utilities"""

    _dht = _collaborative_optimizer = _training_dataset = _authorizer = None

    def __init__(
        self, peer_args: BasePeerArguments, trainer_args: HFTrainerArguments, collab_args: CollaborativeArguments
    ):

        self.peer_args, self.trainer_args, self.collab_args = peer_args, trainer_args, collab_args
        self.validators, self.local_public_key = utils.make_validators(self.peer_args.run_id)

        if self.authorizer:
            self.trainer_args.run_name = self.authorizer.username  # For wandb
        transformers.set_seed(trainer_args.seed)  # seed used for initialization

        self.config = LeanAlbertConfig.from_pretrained(peer_args.model_config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(peer_args.tokenizer_path, cache_dir=peer_args.cache_dir)

        output_dir = Path(trainer_args.output_dir)
        logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
        latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

        if latest_checkpoint_dir is None:
            logger.info(f"Creating model")
            self.model = LeanAlbertForPreTraining(self.config)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            logger.info(f"Loading model from {latest_checkpoint_dir}")
            self.model = LeanAlbertForPreTraining.from_pretrained(latest_checkpoint_dir)
        if trainer_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.current_sequence_length = mp.Value(ctypes.c_int64, self.trainer_args.max_sequence_length)
        self.update_sequence_length()  # updated by callback

    @property
    def authorizer(self):
        if self._authorizer is None and self.peer_args.authorize:
            self._authorizer = authorize_with_huggingface()
        return self._authorizer

    @property
    def dht(self):
        if self._dht is None:
            self._dht = hivemind.DHT(
                start=True,
                initial_peers=self.peer_args.initial_peers,
                client_mode=self.peer_args.client_mode,
                host_maddrs=self.peer_args.host_maddrs,
                announce_maddrs=self.peer_args.announce_maddrs,
                use_ipfs=self.peer_args.use_ipfs,
                record_validators=self.validators,
                identity_path=self.peer_args.identity_path,
                authorizer=self.authorizer,
            )
            if self.peer_args.client_mode:
                logger.info(f"Created client mode peer with peer_id={self._dht.peer_id}")
            else:
                utils.log_visible_maddrs(self._dht.get_visible_maddrs(), only_p2p=self.peer_args.use_ipfs)
        return self._dht

    @property
    def collaborative_optimizer(self) -> hivemind.Optimizer:
        if self._collaborative_optimizer is None:
            averaging_compression = SizeAdaptiveCompression(
                threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization()
            )

            self._collaborative_optimizer = hivemind.Optimizer(
                dht=self.dht,
                params=self._make_param_groups(),
                run_id=self.peer_args.run_id,
                optimizer=self._make_optimizer,
                scheduler=self._make_scheduler,
                grad_compression=averaging_compression,
                state_averaging_compression=averaging_compression,
                batch_size_per_step=self.trainer_args.batch_size_per_step,
                client_mode=self.peer_args.client_mode,
                verbose=True,
                averager_opts=dict(min_vector_size=4_000_000),
                **asdict(self.collab_args),
            )
        return self._collaborative_optimizer

    def _make_param_groups(self) -> ParamGroups:
        no_decay = ["bias", "LayerNorm.weight"]
        return [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.trainer_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def _make_optimizer(self, param_groups: ParamGroups) -> torch.optim.Optimizer:
        return CPULAMB8Bit(
            param_groups,
            lr=self.trainer_args.learning_rate,
            betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
            max_grad_norm=self.trainer_args.max_grad_norm,
            clamp_value=self.trainer_args.clamp_value,
            eps=self.trainer_args.adam_epsilon,
            weight_decay=self.trainer_args.weight_decay,
            reuse_grad_buffers=True,
            bias_correction=True
        )

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> LRSchedulerBase:
        num_warmup_steps = self.trainer_args.warmup_steps
        num_training_steps = self.trainer_args.total_steps
        min_learning_rate = self.trainer_args.min_learning_rate

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            decaying = float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_learning_rate, decaying)

        return LambdaLR(optimizer, lr_lambda)

    @property
    def training_dataset(self):
        if self._training_dataset is None:
            self._training_dataset = make_training_dataset(
                self.tokenizer,
                shuffle_seed=hash(self.local_public_key) % 2 ** 31,
                max_sequence_length=self.current_sequence_length,  # this is a mp.Value that will be changed later
            )
        return self._training_dataset

    def update_sequence_length(self):
        """
        If ramp-up is enabled, start with smaller sequences of initial_sequence_length tokens, then increase linearly
        to the max_sequence_length over the period of first
        """
        current_epoch = self._collaborative_optimizer.tracker.global_epoch
        if self.trainer_args.sequence_length_warmup_steps == 0 or current_epoch > self.trainer_args.sequence_length_warmup_steps:
            current_sequence_length = self.trainer_args.max_sequence_length
        else:
            increment_size = self.trainer_args.pad_to_multiple_of
            max_sequence_length = self.trainer_args.max_sequence_length
            initial_sequence_length = self.trainer_args.initial_sequence_length or increment_size
            sequence_length_warmup_steps = self.trainer_args.sequence_length_warmup_steps
            assert sequence_length_warmup_steps > 0 and max_sequence_length >= initial_sequence_length
            length_range = max_sequence_length - initial_sequence_length
            warmup_relative = min(1, current_epoch / sequence_length_warmup_steps)
            current_sequence_length = initial_sequence_length + warmup_relative * length_range
            current_sequence_length = (current_sequence_length // increment_size) * increment_size
            current_sequence_length = min(max(current_sequence_length, initial_sequence_length), max_sequence_length)

        current_sequence_length = int(current_sequence_length)
        if current_sequence_length != self.current_sequence_length.value:
            logger.info(f"Beginning transition to sequence length {current_sequence_length}")
            self.current_sequence_length.value = current_sequence_length
            # note: it may take time for new sequence length to take effect due to buffering

    @property
    def data_collator(self):
        return DataCollatorForWholeWordMask(
            tokenizer=self.tokenizer, pad_to_multiple_of=self.trainer_args.pad_to_multiple_of
        )
