from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar, Any, Mapping

T = TypeVar("T", bound="GradioConfig")


@dataclass(kw_only=True)
class GradioConfig:
    models_folder: Path = field(
        default="/lustre/fsmisc/dataset/HuggingFace_Models/bigcode",
        metadata={"converter": Path}
    )
    visu_data_path: Path = field(
        default=Path.cwd() / "trl_llm" / "data" / "visual_prompt.json",
        metadata={"converter": Path, "export": False},
    )

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--models_folder",
            "--models-folder",
            "--models_dir",
            "--models-dir",
            dest="models_folder"
        )
        parser.add_argument(
            "--visu-data-path",
            "--visu_data_path",
            dest="visu_data_path"
        )

        return parser

    @classmethod
    def from_mappings(cls: type[T], *mappings: Mapping[str, Any] | Namespace) -> T:
        unified_dict: dict[str, Any] = dict()
        for mapping in mappings:
            if isinstance(mapping, Namespace):
                mapping = vars(mapping)
            unified_dict.update(**mapping)
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in unified_dict.items() if k in class_fields})

    def __str__(self) -> str:
        r"""
        Prints the configuration in a pretty multiline way. Each field will be
        printed on a different line to improve readability.
        """
        string = f"{self.__class__.__qualname__}(\n"
        for f in fields(self):
            string += " " * 4 + f"{f.name}={self.__getattribute__(f.name)},\n"
        string += ")"
        return string
