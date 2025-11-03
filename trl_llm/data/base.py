import random
import re
from abc import ABC, abstractmethod
from dataclasses import field, dataclass

from trl_llm.train.config import TrainingConfig

PATTERN_BACKTIP = r'```.*?```'

@dataclass
class BaseDataset(ABC):
    name: str
    lean_statement: str = field()
    rocq_statement: str = field()

    def get_code(self, config: TrainingConfig) -> tuple[str, str]:
        rocq_code = config.template_rocq.format(content=self.rocq_statement)
        lean_code = config.template_lean.format(content=self.lean_statement)
        return rocq_code, lean_code

    def make_template(self, config: TrainingConfig, eval_template: bool, conversion_type: str) -> str|dict:
        rocq_code, lean_code = self.get_code(config)
        if conversion_type == "lean2rocq":
            template_target = config.template_rocq
            template_conversion = config.template_lean_to_rocq
            source_code = lean_code
            target_code = rocq_code
        elif conversion_type == "rocq2lean":
            template_target = config.template_lean
            template_conversion = config.template_rocq_to_lean
            source_code = rocq_code
            target_code = lean_code
        else:
            raise ValueError("Unknown conversion_type")

        target_part = template_target.split("\n")[0] + "\n" if eval_template else target_code

        prompt = template_conversion + source_code + "\n" + target_part
        completion = re.findall(PATTERN_BACKTIP, target_code, re.DOTALL)[0]  # re.DOTALL affects what the `.` pattern can match. Newlines are matched.

        if config.prompt_type == "sft":
            return prompt
        elif config.prompt_type == "grpo":
            return {"prompt": prompt, "completion": completion}
        else:
            raise ValueError("Unknown prompt_type value.")

    def rocq_to_lean_template(self, config: TrainingConfig, eval_template: bool) -> str|dict:
        return self.make_template(
            config=config,
            eval_template=eval_template,
            conversion_type="rocq2lean"
        )

    def lean_to_rocq_template(self, config: TrainingConfig, eval_template: bool) -> str|dict:
        return self.make_template(
            config=config,
            eval_template=eval_template,
            conversion_type="lean2rocq"
        )

    def get_sample(self, config: TrainingConfig, template: str, eval_template: bool) -> str:
        if template == "rocq_to_lean":
            return self.rocq_to_lean_template(config, eval_template)
        elif template == "lean_to_rocq":
            return self.lean_to_rocq_template(config, eval_template)
        else:
            raise ValueError(f"Invalid template: {template}")

    def get_sample_random_template(self, config: TrainingConfig, eval_template: bool = False) -> str:
        if config.prompt_type == "grpo": eval_template=True
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