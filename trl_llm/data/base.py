import random
from abc import ABC, abstractmethod
from dataclasses import field, dataclass

from trl_llm.train.config import TrainingConfig


@dataclass
class BaseDataset(ABC):
    name: str
    lean_statement: str = field()
    rocq_statement: str = field()

    def get_code(self, config: TrainingConfig) -> tuple[str, str]:
        rocq_code = config.template_rocq.format(content=self.rocq_statement)
        lean_code = config.template_lean.format(content=self.lean_statement)
        return rocq_code, lean_code

    def rocq_to_lean_template(self, config: TrainingConfig, eval_template: bool) -> str:
        rocq_code, lean_code = self.get_code(config)
        lean_part = config.template_lean.split("\n")[0] + "\n" if eval_template else lean_code
        return config.template_rocq_to_lean + rocq_code + "\n" + lean_part

    def lean_to_rocq_template(self, config: TrainingConfig, eval_template: bool) -> str:
        rocq_code, lean_code = self.get_code(config)
        rocq_part = config.template_rocq.split("\n")[0] + "\n" if eval_template else rocq_code
        return config.template_lean_to_rocq + lean_code + "\n" + rocq_part

    def get_sample(self, config: TrainingConfig, template: str, eval_template: bool) -> str:
        if template == "rocq_to_lean":
            return self.rocq_to_lean_template(config, eval_template)
        elif template == "lean_to_rocq":
            return self.lean_to_rocq_template(config, eval_template)
        else:
            raise ValueError(f"Invalid template: {template}")

    def get_sample_random_template(self, config: TrainingConfig, eval_template: bool = False) -> str:
        template_choice = random.choices(
            ["rocq_to_lean", "lean_to_rocq"],
            weights=[
                config.probability_rocq_to_lean,
                1 - config.probability_rocq_to_lean
            ]
        )[0]
        return self.get_sample(config, template_choice, eval_template)

    @classmethod
    @abstractmethod
    def get_data_path(cls, config: TrainingConfig):
       ...

@dataclass
class MiniF2F(BaseDataset):
    lean_header: str = field()
    rocq_header: str = field()
    informal_statement: str = field()
    informal_proof: str = field()

    def get_code(self, config: TrainingConfig) -> tuple[str, str]:
        rocq_code = config.template_rocq.format(content=self.rocq_header + "\n" + self.rocq_statement)
        lean_code = config.template_lean.format(content=self.lean_header + "\n" + self.lean_statement)
        return rocq_code, lean_code

    @classmethod
    def get_data_path(cls, config: TrainingConfig):
        return config.minif2f_data_path

@dataclass
class Putnam(BaseDataset):
    @classmethod
    def get_data_path(cls, config: TrainingConfig):
        return config.putnam_data_path

@dataclass
class Visual(BaseDataset):
    @classmethod
    def get_data_path(cls, config: TrainingConfig):
        pass

@dataclass
class DocMathlibMathcomp:
    filename: str
    content: str

    @classmethod
    def get_data_path(cls, config: TrainingConfig):
        return config.mathlib_mathcomp_data_path

    def get_sample_random_template(self, config: TrainingConfig):
        return self.content