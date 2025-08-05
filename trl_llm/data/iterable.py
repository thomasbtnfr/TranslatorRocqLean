from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset

from trl_llm.data.base import BaseDataset
from trl_llm.train.config import TrainingConfig


@dataclass
class TRLIterableDataset(TorchIterableDataset):
    config: TrainingConfig
    tokenizer: AutoTokenizer
    sample_cls: BaseDataset
    split: str = "train"
    _min = torch.finfo(torch.bfloat16).min

    def __post_init__(self):
        self.dataset = load_dataset(str(self.sample_cls.get_data_path(self.config)), split=self.split)

    def separated_samples_iter(self):
        seed: int = 123456

        while True:
            dataset = self.dataset.shuffle(seed=seed)  # no buffer_size since it is not iterable
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