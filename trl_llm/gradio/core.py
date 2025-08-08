import contextlib
import gc
import os
from pathlib import Path

import gradio as gr
import ray
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

llm = None

def load_model(base_path: Path, chosen_model: str):
    model_path = os.path.join(base_path, chosen_model)
    global llm
    llm = LLM(model_path)
    return "Model loaded"

def unload_model():
    destroy_model_parallel()
    destroy_distributed_environment()
    global llm
    del llm.llm_engine.model_executor
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    return "Model unloaded"


def predict(prompt: str, max_tokens, temperature, presence_penalty, frequency_penalty, top_p, top_k) -> str:
    if len(prompt) == 0:
        raise gr.Error("You can't submit an empty request")

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        top_p=top_p,
        top_k=top_k,
    )

    result = llm.generate(prompt, sampling_params)
    return result[0].outputs[0].text