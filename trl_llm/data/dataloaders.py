from itertools import islice

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from trl_llm.data.base import MiniF2F, Putnam, DocMathlibMathcomp
from trl_llm.data.iterable import TrainTRLIterableDataset, ValidTRLIterableDataset
from trl_llm.train.config import TrainingConfig


def get_dataloaders(config: TrainingConfig, tokenizer: AutoTokenizer):
    if "minif2f" in config.exp_name.lower():
        train_dataset = TrainTRLIterableDataset(
            config=config,
            tokenizer=tokenizer,
            sample_cls=MiniF2F,
            split="valid"
        )
        val_dataset = ValidTRLIterableDataset(
            config=config,
            tokenizer=tokenizer,
            sample_cls=MiniF2F,
            split="test"
        )
    elif "putnam" in config.exp_name.lower():
        train_dataset = TrainTRLIterableDataset(
            config=config,
            tokenizer=tokenizer,
            sample_cls=Putnam,
            split="train"
        )
        val_dataset = ValidTRLIterableDataset(
            config=config,
            tokenizer=tokenizer,
            sample_cls=Putnam,
            split="train"  # TODO: prepare valid dataset
        )
    elif any(word in config.exp_name.lower() for word in ["mathlib", "mathcomp", "doc"]):
        train_dataset = TrainTRLIterableDataset(
            config=config,
            tokenizer=tokenizer,
            sample_cls=DocMathlibMathcomp,
            split="train"
        )
        val_dataset = ValidTRLIterableDataset(
            config=config,
            tokenizer=tokenizer,
            sample_cls=DocMathlibMathcomp,
            split="validation"
        )
    else:
        raise ValueError("exp_name value does not permit to load one of the datasets.")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=10,
        persistent_workers=True
    )

    if config.resume_from_step > 0:
        train_loader = islice(
            train_loader,
            config.resume_from_step * config.gradient_accumulation,
            None,
        )
    return train_loader, val_loader