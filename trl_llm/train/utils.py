import warnings
from abc import abstractmethod
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Callable, cast

import idr_torch
import mlflow
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FSDPModule, fully_shard, register_fsdp_forward_method
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_post_forward_mesh_info
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ConstantLR, LinearLR, LRScheduler, SequentialLR
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

from trl_llm.data.visual import VisuDataset
from trl_llm.train.config import TrainingConfig

warnings.filterwarnings("ignore", category=SyntaxWarning)


class ModelForCausalLM(PreTrainedModel, GenerationMixin):

    @abstractmethod
    def set_reshard_after_forward(self, reshard_after_forward: bool) -> None:
        ...


def init_mlflow(config: TrainingConfig):
    # Assure that every process has the same config.run_name
    if idr_torch.world_size > 1:
        dist.barrier(device_ids=[idr_torch.local_rank])
        objects = [config.run_name]
        dist.broadcast_object_list(objects, src=0, device=idr_torch.device)
        config.run_name = objects[0]

    if config.should_log:
        mlflow.set_tracking_uri("file://" + str(config.mlflow_path))
        mlflow.set_experiment(config.exp_name)

        if mlflow.search_runs(filter_string=f"run_name='{config.run_name}'").empty:
            run_id = None
        else:
            run_id = mlflow.search_runs(
                filter_string=f"run_name='{config.run_name}'"
            ).iloc[0]["run_id"]
        return mlflow.start_run(run_name=config.run_name, run_id=run_id)  # as _:
    else:
        return nullcontext()


def cleanup():
    dist.barrier(device_ids=[idr_torch.local_rank])
    dist.destroy_process_group()


def get_optimizer(config: TrainingConfig, model: torch.nn.Module) -> Optimizer:
    return AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),  # (0.95, 0.995)
    )


def get_lr_scheduler(
    optimizer: Optimizer, warmup_steps: int, num_steps: int
) -> LRScheduler:
    assert num_steps > 2 * warmup_steps

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-5,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    constant_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=num_steps - 2 * warmup_steps,
    )
    annealing_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1e-5,
        total_iters=warmup_steps,
    )
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler, annealing_scheduler],
        milestones=[warmup_steps, num_steps - warmup_steps],
    )
    return lr_scheduler


def get_criterion(
    pad_token_id: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    criterion = CrossEntropyLoss(ignore_index=pad_token_id)

    def wrapped(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, latent_dim = predictions.shape
        return criterion(
            predictions.view(batch_size * seq_len, latent_dim),
            labels.view(batch_size * seq_len),
        )

    return wrapped


def get_profiler(path: Path, enable: bool = True):
    if enable:
        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=5, warmup=5, active=20, repeat=1),
            on_trace_ready=tensorboard_trace_handler(str(path)),
            profile_memory=True,
            with_stack=False,
            record_shapes=False,
        )
    else:
        context = nullcontext()
        context.step = lambda *args, **kwargs: None
        return context

def visual_prompt(
    visu_dataset: VisuDataset,
    config: TrainingConfig,
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
):
    """Prints generated text and target text"""
    if not config.deactivate_sharding:
        model.set_reshard_after_forward(False)
    inp_prompts = [visu_dataset[idx] for idx in range(len(visu_dataset))]
    inputs = tokenizer(
        inp_prompts, truncation=False, padding=True, return_tensors="pt"
    ).to(idr_torch.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,
        max_new_tokens=500,
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
    )

    input_tokens_lengths = [x.shape[0] for x in inputs["input_ids"]]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [
        o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
    ]

    outputs_new = []
    for i, total_new_token in enumerate(total_new_tokens):
        outputs_new.append(
            tokenizer.batch_decode(
                [outputs[i][-total_new_token:]], skip_special_tokens=True
            )[0]
        )

    if config.track:
        step_note = [f"Step {config.step}:"] * len(inp_prompts)
        table_dict = {
            "step": step_note,
            "prompts": inp_prompts,
            "outputs": outputs_new,
        }
        mlflow.log_table(data=table_dict, artifact_file="visual_prompt.json")
    if not config.deactivate_sharding:
        model.set_reshard_after_forward(True)

class FullAppState(Stateful):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        ntokens: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ntokens = ntokens

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "ntokens": self.ntokens,
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.ntokens = state_dict["ntokens"]


class ModelAppState(Stateful):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict = get_model_state_dict(self.model)
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_model_state_dict(self.model, model_state_dict=state_dict["model"])


def save(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    lr_scheduler: LRScheduler | None = None,
    ntokens: int | None = None,
) -> None:
    if idr_torch.rank == 0:
        print('Saving model.')
    if optimizer is not None and lr_scheduler is not None and ntokens is not None:
        state_dict = {"app": FullAppState(model, optimizer, lr_scheduler, ntokens)}
    else:
        state_dict = {"app": ModelAppState(model)}
    dcp.save(state_dict=state_dict, checkpoint_id=path)


def load(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    lr_scheduler: LRScheduler | None = None,
) -> int:
    if optimizer is not None and lr_scheduler is not None:
        appState = FullAppState(model, optimizer, lr_scheduler)
    else:
        appState = ModelAppState(model)
    state_dict = {"app": appState}
    dcp.load(state_dict=state_dict, checkpoint_id=path)
    return appState.ntokens if isinstance(appState, FullAppState) else 0


def sync(src_model: torch.nn.Module, dst_model: torch.nn.Module) -> None:
    set_model_state_dict(
        dst_model,
        get_model_state_dict(src_model)
    )


def set_reshard_after_forward(self, reshard_after_forward: bool) -> None:
    """
    See https://github.com/pytorch/pytorch/issues/149029
    and https://github.com/pytorch/pytorch/pull/149103

    Sets if the module should reshard parameters after forward. This can be
    used to change the ``reshard_after_forward`` FSDP arg at runtime. For
    example, it can set an FSDP module's value to ``False`` for running evals
    and set back to ``True`` for training.

    Args:
        reshard_after_forward (bool): Whether to reshard parameters after
            forward.
    """
    self_module = cast(nn.Module, self)
    modules = [module for module in self_module.modules() if module is not self_module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.post_forward_mesh_info = (
                    _get_post_forward_mesh_info(
                        reshard_after_forward, fsdp_param_group.mesh_info
                    )
                )
            if reshard_after_forward:
                module.reshard()


def shard(model: AutoModelForCausalLM, hsdp: bool = True):
    if hsdp:
        mesh_shape = (idr_torch.nnodes // 2, 2 * idr_torch.local_world_size)
    else:
        mesh_shape = (1, idr_torch.world_size)
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=mesh_shape,
        mesh_dim_names=("replicate", "shard"),
    )

    # Apply HSDP
    sharded_classes = (model.model.layers[0].__class__, model.__class__)
    # We reverse the traversal of modules because we want to shard inner modules
    # first and only then shard the outer modules.

    for submodule in reversed(list(model.modules())):
        if isinstance(submodule, sharded_classes):
            fully_shard(submodule, mesh=mesh, reshard_after_forward=True)
    register_fsdp_forward_method(model, "generate")  # to use generate with FSDP2
    setattr(
        model,
        "set_reshard_after_forward",
        MethodType(set_reshard_after_forward, model),
    )
    return model


def init_sharded_model(config: TrainingConfig) -> ModelForCausalLM:
    model: ModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map=idr_torch.device,
        trust_remote_code=True,
        _attn_implementation="sdpa",
    )
    if not config.deactivate_sharding:
        model = shard(model)
    return model


def init_model_training(
    config: TrainingConfig,
) -> tuple[ModelForCausalLM, PreTrainedTokenizer, Optimizer, LRScheduler, int]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_id = -1000
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if config.resume_from_step > 0:
        # Resume from checkpoint
        model = init_sharded_model(config)
        optimizer = get_optimizer(config, model)
        lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=config.warmup_steps, num_steps=config.num_steps)
        ntokens = load(config.checkpoint_path, model, optimizer, lr_scheduler)

    elif False and len(list(config.model_path.glob("__*_0.distcp"))) > 0:
        # Loading foundation model from DCP is much faster
        model = init_sharded_model(config)
        load(config.model_path, model)
        optimizer = get_optimizer(config, model)
        lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=config.warmup_steps, num_steps=config.num_steps)
        ntokens = 0

    else:
        model = init_sharded_model(config)
        optimizer = get_optimizer(config, model)
        lr_scheduler = get_lr_scheduler(
            optimizer, warmup_steps=config.warmup_steps, num_steps=config.num_steps
        )
        ntokens = 0

    return model, tokenizer, optimizer, lr_scheduler, ntokens
