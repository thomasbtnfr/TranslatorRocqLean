import time
from argparse import Namespace, ArgumentParser, BooleanOptionalAction
from dataclasses import field, dataclass, fields
from pathlib import Path
from typing import Callable, Any, Mapping, TypeVar

import idr_torch

T = TypeVar("T", bound="TrainingConfig")


@dataclass
class TrainingConfig:
    config_file: str | None = field(
        default=None, metadata={"converter": str, "export": False}
    )
    model_dir: Path = field(
        default=Path.cwd() / "model", metadata={"converter": Path, "export": False}
    )
    model_name: str = field(
        default="DeepSeek-Coder-V2-Lite-Instruct", metadata={"converter": str, "export": True},
    )
    minif2f_data_path: Path = field(
        default=None, metadata={"converter": Path, "export": True}
    )
    putnam_data_path: Path = field(
        default=None, metadata={"converter": Path, "export": True}
    )
    mathlib_mathcomp_data_path: Path = field(
        default=None, metadata={"converter": Path, "export": True}
    )
    checkpoints_dir: Path = field(
        default=Path.cwd() / "checkpoints", metadata={"converter": Path, "export": True}
    )
    profiler_path: Path = field(
        default=Path.cwd() / "profiler", metadata={"converter": Path, "export": False}
    )
    mlflow_path: Path = field(
        default=Path.cwd() / "mlruns", metadata={"converter": Path, "export": False}
    )
    visu_data_path: Path = field(
        default=Path.cwd() / "trl_llm" / "data" / "visual_prompt.json"
    )
    visual_prompt_generated_path: Path = field(
        default=None, metadata={"converter": Path, "export": False}
    )
    compile_lean_dir_path: Path = field(
        default=None, metadata={"converter": Path, "export": False}
    )
    compile_lean_file_path: Path = field(
        default="CompileLean.lean", metadata={"converter": Path, "export": False}
    )
    dcp_to_hf_saved_name: str = field(
        default="starcoder2-instruct", metadata={"converter": str, "export": False}
    )
    exp_name: str = field(default="test", metadata={"converter": str, "export": False})
    run_name: str = field(default="", metadata={"converter": str, "export": False})
    seq_length: int = field(default=2048, metadata={"converter": int, "export": True})
    profile: bool = field(default=False, metadata={"converter": bool, "export": False})
    _track: bool = field(default=False, metadata={"converter": bool, "export": False})
    debug: bool = field(default=False, metadata={"converter": bool, "export": True})
    visual_prompt: bool = field(default=False, metadata={"converter": bool, "export": False})
    deactivate_sharding: bool = field(default=False, metadata={"converter": bool, "export": True})
    do_not_save: bool = field(default=False, metadata={"converter": bool, "export": True})
    use_putnam_visual: bool = field(default=False, metadata={"converter": bool, "export": True})
    num_steps: int = field(default=1000, metadata={"converter": int, "export": True})
    lr: float = field(default=1e-05, metadata={"converter": float, "export": True})
    warmup_steps: int = field(default=200, metadata={"converter": int, "export": True})
    batch_size: int = field(default=1, metadata={"converter": int, "export": True})
    eval_steps: int = field(default=1000, metadata={"converter": int, "export": True})
    visual_n_first_prompt: int = field(default=-1, metadata={"converter": int, "export": True})
    template_rocq: str = field(
        default="Rocq code:\n```rocq\n{content}\n```",
        metadata={"converter": str, "export": True}
    )
    template_lean: str = field(
        default="Lean code:\n```lean\n{content}\n```",
        metadata={"converter": str, "export": True}
    )
    template_rocq_to_lean: str = field(
        default="Convert Rocq code to Lean code.\n",
        metadata={"converter": str, "export": True}
    )
    template_lean_to_rocq: str = field(
        default="Convert Lean code to Rocq code.\n",
        metadata={"converter": str, "export": True}
    )
    probability_rocq_to_lean: float = field(
        default=0.5, metadata={"converter": float, "export": True}
    )
    resume_from_step: int = field(
        default=0, metadata={"converter": int, "export": False}
    )
    gradient_accumulation: int = field(
        default=1, metadata={"converter": int, "export": True}
    )

    step: int = field(default=0, init=False)
    seed: int = field(default=53)

    def __post_init__(self) -> None:
        for dataclass_field in fields(self):
            converter: Callable[[Any], Any] | None = dataclass_field.metadata.get(
                "converter", None
            )
            if converter is not None:
                value = getattr(self, dataclass_field.name)
                if value is not None:
                    self.__setattr__(dataclass_field.name, converter(value))

        if self.run_name == "":
            self.run_name = f"{self.exp_name.lower()}-{time.time_ns()}"

        self.step = self.resume_from_step

    @classmethod
    def from_mappings(cls: type[T], *mappings: Mapping[str, Any] | Namespace) -> T:
        unified_dict: dict[str, Any] = dict()
        for mapping in mappings:
            if isinstance(mapping, Namespace):
                mapping = vars(mapping)
            unified_dict.update(**mapping)
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in unified_dict.items() if k in class_fields})

    def export(self, *, full: bool = False) -> dict[str, Any]:
        all_fields = fields(self)
        all_exportable_fields = [f.name for f in all_fields if f.metadata.get("export")]
        filtered_export = dict()
        for key, value in vars(self).items():
            if full or key in all_exportable_fields:
                filtered_export[key] = (
                    value if not isinstance(value, Path) else str(value)
                )
        return filtered_export

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--model-dir", "--model_dir", dest="model_dir")
        parser.add_argument("--model-name", "--model_name", dest="model_name")
        parser.add_argument(
            "--checkpoints-dir", "--checkpoints_dir", dest="checkpoints_dir"
        )
        parser.add_argument(
            "--minif2f-data-path", "--minif2f_data_path", "--minif2f-path", "--minif2f_path", dest="minif2f_data_path"
        )
        parser.add_argument(
            "--putnam-data-path", "--putnam_data_path", "--putnam-path", "--putnam_path", dest="putnam_data_path"
        )
        parser.add_argument(
            "--mathlib-mathcomp-data-path", "--mathlibmathcomp", "--mathlib_mathcomp_data_path",
            "--mathlibmathcomp_path", "--mathlibmathcomp-path", dest="mathlib_mathcomp_data_path"
        )
        parser.add_argument(
            "--mlflow_path", "--mlflow-path", dest="mlflow_path"
        )
        parser.add_argument(
            "--visu-data-path",
            "--visu_data_path",
            dest="visu_data_path"
        )
        parser.add_argument(
            "--visual-prompt-generated-path",
            "--visual_prompt_generated_path",
            dest="visual_prompt_generated_path"
        )
        parser.add_argument(
            "--compile_lean_dir_path",
            "--compile-lean-dir-path",
            dest="compile_lean_dir_path"
        )
        parser.add_argument(
            "--compile_lean_file_path",
            "--compile-lean-file-path",
            dest="compile_lean_file_path"
        )
        parser.add_argument(
            "--dcp-to-hf-saved-name",
            "--dcp_to_hf_saved_name",
            dest="dcp_to_hf_saved_name"
        )
        parser.add_argument("--exp-name", "--exp_name", dest="exp_name")
        parser.add_argument("--run-name", "--run_name", dest="run_name")
        parser.add_argument(
            "--sequence-length", "--sequence_length", "--seq-length", "--seq_length", dest="seq_length",
            help="Length of the sequence",
        )
        parser.add_argument("--steps", "--num_steps", "--num-steps", dest="num_steps")
        parser.add_argument("--lr")
        parser.add_argument("--profile", action=BooleanOptionalAction)
        parser.add_argument("--track", dest="_track", action=BooleanOptionalAction)
        parser.add_argument("--debug", action=BooleanOptionalAction)
        parser.add_argument("--visual-prompt", action=BooleanOptionalAction)
        parser.add_argument("--deactivate-sharding", action=BooleanOptionalAction)
        parser.add_argument("--do-not-save", action=BooleanOptionalAction)
        parser.add_argument("--use-putnam-visual", action=BooleanOptionalAction)
        parser.add_argument("--warmup_steps", "--warmup-steps", dest="warmup_steps")
        parser.add_argument("--bsz", "--batch-size", "--batch_size", dest="batch_size")
        parser.add_argument("--eval-steps", "--eval_steps", dest="eval_steps")
        parser.add_argument("--template-rocq", "--template_rocq", dest="template_rocq")
        parser.add_argument("--template-lean", "--template_lean", dest="template_lean")
        parser.add_argument("--template-rocq-to-lean", "--template_rocq_to_lean", dest="template_rocq_to_lean")
        parser.add_argument("--template-lean-to-rocq", "--template_lean_to_rocq", dest="template_lean_to_rocq")
        parser.add_argument("--prob", "--probability_rocq_to_lean", "--probability-rocq-to-lean", dest="probability_rocq_to_lean")
        parser.add_argument("--grad-acc", "--grad_acc", dest="gradient_accumulation")
        parser.add_argument("--resume-from-step", "--resume_from_step", dest="resume_from_step")
        parser.add_argument("--visual-n-first-prompt", dest="--visual_n_first_prompt")
        return parser


    @property
    def should_log(self) -> bool:
        return self._track and idr_torch.rank == 0

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name

    @property
    def save_path(self) -> Path:
        return self.checkpoints_dir / self.exp_name / self.run_name

    @property
    def checkpoint_path(self) -> Path:
        return self.save_path / f"step_{self.step}"

    @property
    def track(self) -> bool:
        return self._track and idr_torch.rank == 0

    @property
    def compile_lean_full_file_path(self) -> Path:
        return self.compile_lean_dir_path / self.compile_lean_file_path

    def __str__(self) -> str:
        """
        Prints the configuration in a pretty multiline way. Each field will be
        printed on a different line to improve readability.
        """
        string = f"{self.__class__.__qualname__}(\n"
        for f in fields(self):
            string += " " * 4 + f"{f.name}={self.__getattribute__(f.name)},\n"
        string += ")"
        return string