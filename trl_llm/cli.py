import argparse
import configparser
import json
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path
from typing import Any, Protocol, cast

import yaml

from trl_llm.gradio.config import GradioConfig
from trl_llm.train.config import TrainingConfig


def get_cfg_file_args(config_file: Path | None) -> dict[str, Any]:
    if config_file is None:
        return dict()

    with config_file.open("r") as cfg_file:
        match config_file.suffix:
            case ".json":
                defaults = json.load(cfg_file)
            case ".yaml":
                defaults = yaml.safe_load(cfg_file)
            case ".ini" | ".cfg":
                cfg = configparser.ConfigParser()
                cfg.read_file(cfg_file)
                defaults = {s: dict(cfg.items(s)) for s in cfg.sections()}
            case _:
                raise NotImplementedError("This file type is not acceptable.")
    return defaults

class BaseNamespaceProtocol(Protocol):
    action: str
    config_file: Path


class BaseNamespace(argparse.Namespace, BaseNamespaceProtocol):
    pass

def cli():
    parser = ArgumentParser("Finetuning LLM for bottom-up code generation")

    subparsers = parser.add_subparsers(dest="action")

    config_file_parser = ArgumentParser(add_help=False)
    config_file_parser.add_argument(
        "-c",
        "--config",
        "--configfile",
        "--config_file",
        "--config-file",
        dest="config_file",
        type=Path,
        default=None,
    )

    training_parser = subparsers.add_parser(
        "train",
        argument_default=SUPPRESS,
        parents=[config_file_parser]
    )
    TrainingConfig.add_args(training_parser)

    gradio_parser = subparsers.add_parser(
        "gradio",
        argument_default=SUPPRESS,
        parents=[config_file_parser]
    )
    GradioConfig.add_args(gradio_parser)

    dcp_to_hf = subparsers.add_parser(
        "dcp2hf",
        argument_default=SUPPRESS,
        parents=[config_file_parser]
    )
    TrainingConfig.add_args(dcp_to_hf)

    grpo_parser = subparsers.add_parser(
        "grpo",
        argument_default=SUPPRESS,
        parents=[config_file_parser]
    )
    TrainingConfig.add_args(grpo_parser)

    visual_prompt_compile = subparsers.add_parser(
        "compile_visual_prompt",
        argument_default=SUPPRESS,
        parents=[config_file_parser]
    )
    TrainingConfig.add_args(visual_prompt_compile)

    cli_args = cast(BaseNamespace, parser.parse_args())
    cfg_file_args = get_cfg_file_args(cli_args.config_file)

    if cli_args.action == "train":
        from trl_llm.train.trainer import train

        config = TrainingConfig.from_mappings(cfg_file_args, cli_args)
        train(config)
    elif cli_args.action == "grpo":
        from trl_llm.train.grpo import finetune

        config = TrainingConfig.from_mappings(cfg_file_args, cli_args)
        finetune(config)
    elif cli_args.action == "gradio":
        from trl_llm.gradio.main import launch_gradio, make_gradio

        config = GradioConfig.from_mappings(cfg_file_args, cli_args)

        demo = make_gradio(config)
        launch_gradio(demo=demo, port=7886)
    elif cli_args.action == "dcp2hf":
        from trl_llm.train.dcp_to_hf import convert

        config = TrainingConfig.from_mappings(cfg_file_args, cli_args)
        convert(config)
    elif cli_args.action == "compile_visual_prompt":
        from trl_llm.eval.compile_visual_prompt import compile_rocq

        config = TrainingConfig.from_mappings(cfg_file_args, cli_args)
        compile_rocq(config)
    else:
        raise ValueError("Unknown action: {cli_args.action}")


if __name__ == "__main__":
    cli()