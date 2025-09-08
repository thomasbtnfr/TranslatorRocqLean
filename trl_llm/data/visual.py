import json
from functools import partial
from pathlib import Path

from datasets import Dataset as DatasetHF
from torch.utils.data import Dataset

from trl_llm.data.base import Visual
from trl_llm.data.iterable import get_shard_arrow_dataset
from trl_llm.train.config import TrainingConfig


class VisuDataset(Dataset):
    def __init__(self, config: TrainingConfig):
        self.config = config

        if not config.use_putnam_visual:
            with open(config.visu_data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            path = Path(f"{self.config.putnam_data_path}/valid")
            self.data = get_shard_arrow_dataset(dataset_path=path)
            # Convert IterableDataset to Dataset
            self.data = DatasetHF.from_list(list(self.data), features=self.data.features)

        self.data = self.data.select(range(self.config.visual_n_first_prompt)) \
            if self.config.visual_n_first_prompt != -1 else self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        sample = self.data[idx]

        sample_ds = Visual(
            name="",
            lean_statement=sample["lean_statement"],
            rocq_statement=sample["rocq_statement"]
        )
        return sample_ds.get_sample_random_template(config=self.config, eval_template=True)