import os
import time
from datetime import timedelta

import idr_torch
import mlflow
import torch
from torchmetrics.aggregation import RunningMean, SumMetric

from trl_llm.data.dataloaders import get_dataloaders
from trl_llm.data.visual import VisuDataset
from trl_llm.train.config import TrainingConfig
from trl_llm.train.utils import init_mlflow, get_profiler, init_model_training, get_criterion, cleanup, save, \
    visual_prompt
from trl_llm.eval.evaluate import evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(config: TrainingConfig):
    torch.cuda.set_device(idr_torch.local_rank)
    device = idr_torch.init_process_group("nccl")
    mlflow_context = init_mlflow(config)
    profiler = get_profiler(config.profiler_path, enable=config.profile)
    if idr_torch.rank == 0:
        print(config)
    model, tokenizer, optimizer, lr_scheduler, ntokens = init_model_training(config)
    train_loader, val_loader = get_dataloaders(config, tokenizer)

    criterion = get_criterion(tokenizer.pad_id)
    loss_avg_metric = RunningMean(window=config.gradient_accumulation).to(device)
    num_tokens = SumMetric().to(device)
    if idr_torch.rank == 0:
        print(model)
        num_tokens.update(ntokens)

    evaluate(config, model, tokenizer, val_loader, device, config.step)
    if config.visual_prompt:
      visu_dataset = VisuDataset(config=config)
      visual_prompt(
          visu_dataset=visu_dataset,
          config=config,
          model=model,
          tokenizer=tokenizer
      )

    with mlflow_context, profiler:
        timestamp = time.perf_counter()
        model.train()
        if config.track:
            mlflow.log_params(config.export())
            mlflow.log_param('world_size', idr_torch.world_size)

        for i, (input_ids, attention_mask, labels) in enumerate(
            train_loader, start=config.step * config.gradient_accumulation + 1
        ):
            assert attention_mask.ndim == 4
            num_tokens.update(input_ids.shape[0] * input_ids.shape[1])
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits

            loss = criterion(logits, labels)
            (loss / config.gradient_accumulation).backward()
            loss_avg_metric.update(loss.detach())

            if i % config.gradient_accumulation == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                config.step += 1
                ntokens = num_tokens.compute().item()
                loss_avg = loss_avg_metric.compute().item()
                if config.track:
                    mlflow.log_metrics(
                        {
                            "train_loss": loss_avg,
                            "train_tokens": ntokens,
                            "learning_rate": lr_scheduler.get_last_lr()[0]
                        },
                        step=config.step,
                    )
                    mlflow.log_metrics(
                        {
                            "train_loss_wrt_tokens": loss_avg,
                        },
                        step=int(ntokens),
                    )
                    mlflow.log_metrics(
                        {
                            "probability_rocq_to_lean": config.probability_rocq_to_lean,
                        },
                        step=config.step
                    )

                if config.step % 20 == 0:
                    if idr_torch.rank == 0:
                        walltime = timedelta(seconds=int(time.perf_counter() - timestamp))
                        print(
                            f"Total optimization steps: {config.step} | Num microbatches: {i} | Loss: {loss_avg:.3f} | LR: {lr_scheduler.get_last_lr()[0]:.1e} | Memory: {torch.cuda.max_memory_allocated() / (1024 ** 3)} | Walltime: {walltime}"
                        )

                if config.step % config.eval_steps == 0:
                    evaluate(config, model, tokenizer, val_loader, device, config.step)
                    if config.visual_prompt:
                        visual_prompt(
                            visu_dataset=visu_dataset,
                            config=config,
                            model=model,
                            tokenizer=tokenizer
                        )
                    # if not config.do_not_save:
                    #     save(config.checkpoint_path, model, optimizer, lr_scheduler, ntokens)

            profiler.step()

            if config.step >= config.num_steps or (config.debug and i % 100 == 0):
                break

    cleanup()
