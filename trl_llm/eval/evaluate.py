import idr_torch
import mlflow
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric
from torchmetrics.text import Perplexity
from transformers import AutoTokenizer

from trl_llm.train.config import TrainingConfig
from trl_llm.train.utils import get_criterion, ModelForCausalLM


def evaluate(
    config: TrainingConfig,
    model: ModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    device: torch.device,
    step: int,
) -> None:
    if not config.deactivate_sharding:
        model.set_reshard_after_forward(False)
    perplexity = Perplexity(ignore_index=tokenizer.pad_id).to(device)
    mean = MeanMetric().to(device)
    done: bool = False
    L = [done] * idr_torch.world_size
    model.eval()
    criterion = get_criterion(pad_token_id=tokenizer.pad_id)

    with torch.no_grad():
        for i, (input_ids, attention_mask, labels) in enumerate(dataloader, start=1):
            assert attention_mask.ndim == 4
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            if not done and (labels == tokenizer.pad_id).all():
                done = True

            if not done:
                loss = criterion(logits, labels)
                mean.update(loss)
                perplexity.update(logits, labels)

            dist.all_gather_object(L, done)
            if all(L):
                break
            if idr_torch.rank == 0 and i % 100 == 0:
                print(f"Evaluation step {i}")

    perplexity = perplexity.compute().item()
    mean = mean.compute().item()

    if config.track:
        mlflow.log_metrics(
            {"perplexity": perplexity, "eval_loss": mean},
            step=step,
        )

    if idr_torch.rank == 0:
        print(f"Perplexity {perplexity:.3f}, Loss {mean:.3f}")

    model.train()
    if not config.deactivate_sharding:
        model.set_reshard_after_forward(True)