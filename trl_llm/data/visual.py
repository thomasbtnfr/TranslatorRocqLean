import json

from torch.utils.data import Dataset

from trl_llm.data.base import Visual
from trl_llm.train.config import TrainingConfig



class VisuDataset(Dataset):
    def __init__(self, config: TrainingConfig):
        with open(config.visu_data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.config = config

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