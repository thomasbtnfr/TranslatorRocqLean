from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import idr_torch
import torch
from datasets import IterableDataset as HuggingFaceDataset
from datasets import load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info
from transformers import AutoTokenizer

from trl_llm.data.base import BaseDataset
from trl_llm.train.config import TrainingConfig


@dataclass
class TrainTRLIterableDataset(TorchIterableDataset):
    config: TrainingConfig
    tokenizer: AutoTokenizer
    sample_cls: BaseDataset
    split: str = "train"
    _min = torch.finfo(torch.bfloat16).min

    def __post_init__(self):
        if "shard" in str(self.sample_cls.get_data_path(self.config)):
            self.dataset = self.get_shard_dataset(dataset_path=Path(
                f"{str(self.sample_cls.get_data_path(self.config))}/{self.split}"))
        else:
            self.dataset = load_dataset(str(self.sample_cls.get_data_path(self.config)), split=self.split)

    def get_shard_dataset(self, dataset_path: Path) -> HuggingFaceDataset:
        arrow_files = sorted(str(f) for f in dataset_path.glob("shard_*/**/*.arrow"))
        sharded_arrow_files = arrow_files[idr_torch.rank:: idr_torch.world_size]
        dataset = load_dataset(
            "arrow",
            data_files=sharded_arrow_files,
            streaming=True,
            split="train",
        )
        return dataset

    def separated_samples_iter(self):
        seed: int = 123456

        while True:
            dataset = self.dataset.shuffle(seed=seed,
                                           buffer_size=100 if isinstance(self.dataset, HuggingFaceDataset) else None)
            seed += 1

            for i, sample in enumerate(dataset):
                yield self.sample_cls(**sample).get_sample_random_template(self.config)

    def tokenize(self, sample: str) -> list[int]:
        return self.tokenizer(sample, return_attention_mask=False)["input_ids"]

    def invert_causal_mask(self, causal_mask: torch.Tensor) -> torch.Tensor:
        causal_mask = causal_mask.unsqueeze(0).to(dtype=torch.bfloat16)
        inverted_causal_mask = (1 - causal_mask) * self._min
        return inverted_causal_mask

    def __iter__(self):
        buf_inputs = []
        buf_masks = []

        for sample in self.separated_samples_iter():
            input_ids = self.tokenize(sample) + [self.tokenizer.eos_token_id]

            if len(input_ids) > self.config.seq_length + 1:
                input_ids = input_ids[:self.config.seq_length + 1]

            causal_mask = torch.tril(torch.ones(len(input_ids), len(input_ids)))

            if len(buf_inputs) + len(input_ids) <= self.config.seq_length + 1:
                # The input fits entirely in the buffer.
                buf_inputs += input_ids
                buf_masks.append(causal_mask)
            else:
                # Pack the data, the mask and the target for a training step.
                data = torch.tensor(buf_inputs)[:-1]
                target = torch.tensor(buf_inputs)[1:]
                attention_mask = torch.block_diag(*buf_masks)
                attention_mask = self.invert_causal_mask(attention_mask[:-1, :-1])

                buf_inputs = input_ids
                buf_masks = [causal_mask]
                yield data, attention_mask, target


class ValidTRLIterableDataset(TrainTRLIterableDataset):
    def separated_samples_iter(self):
        for i, sample in enumerate(self.dataset):
            yield self.sample_cls(**sample).get_sample_random_template(self.config)

    def yield_fake(self):
        # Yield fake data so that during evaluation, even if a rank has fewer samples,
        # it still does a forward so that we can unshard the FSDP model.
        # It is up to the evaluation loop to detect that and break the dataloader loop.
        # Should not happen during training since the dataset is so large.
        fake_data = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        fake_labels = torch.tensor([self.tokenizer.pad_id], dtype=torch.long)
        fake_attn_mask = torch.zeros((1, 1))
        fake_attn_mask = self.invert_causal_mask(fake_attn_mask)
        return repeat((fake_data, fake_attn_mask, fake_labels))

    def __iter__(self):
        for sample in self.separated_samples_iter():
            input_ids = self.tokenize(sample) + [self.tokenizer.eos_token_id]

            if len(input_ids) > self.config.seq_length + 1:
                input_ids = input_ids[:self.config.seq_length + 1]

            data = torch.tensor(input_ids)[:-1]
            target = torch.tensor(input_ids)[1:]
            attention_mask = (
                torch.tril(torch.ones(len(input_ids), len(input_ids)))[:-1, :-1]
            )
            attention_mask = self.invert_causal_mask(attention_mask)
            if attention_mask.ndim == 3:
                yield data, attention_mask, target

        yield from self.yield_fake()
