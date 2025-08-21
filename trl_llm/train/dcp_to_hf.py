import idr_torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_llm.train.config import TrainingConfig
from trl_llm.train.utils import ModelForCausalLM, load


def convert(config: TrainingConfig) -> None:
    print("init...")
    model: ModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map=idr_torch.device,
        trust_remote_code=True,
        _attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    print("load checkpoint...")
    assert config.resume_from_step > 0
    load(config.checkpoint_path, model)

    print("saving...")
    model_name = "starcoder2-instruct"
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)