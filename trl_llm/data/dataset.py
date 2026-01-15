from dataclasses import dataclass

from datasets import load_from_disk
from torch.utils.data import Dataset

from trl_llm.data.base import Putnam
from trl_llm.train.config import TrainingConfig


@dataclass
class GRPOCodeDataset(Dataset):
    config: TrainingConfig
    split: str = "train"

    def __post_init__(self):
        self.dataset = load_from_disk(
            Putnam.get_data_path(self.config),
        )[self.split]

    def __getitem__(self, idx):
        return Putnam(**self.dataset[idx]).get_sample_random_template(self.config)

    def __len__(self):
        return len(self.dataset)