import os
import random
import mlflow
from functools import partial

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from trl_llm.data.dataset import GRPOCodeDataset
from trl_llm.eval.utils import extract_code, check_rocq_code, check_lean_code
from trl_llm.train.config import TrainingConfig



def compile_rewards(completions: list[str], **kwargs) -> list[float]:
    extracted_codes = [extract_code(content) for content in completions]
    config = kwargs.get('config')

    def compile_sample(sample):
        if not sample:
            return 0.0

        language, code = sample
        if not code:
            return 0.0

        if language in {"rocq", "coq"}:
            has_compiled, _ = check_rocq_code(code)
        elif language == "lean":
            has_compiled, _ = check_lean_code(code, config)
        else:
            has_compiled = False

        return 1.0 if has_compiled else 0.0

    return [compile_sample(sample) for sample in extracted_codes]


def template_rewards(completions: list[str], **kwargs) -> list[float]:
    extracted_codes = [extract_code(content) for content in completions]

    verbose = kwargs.get('verbose', False)
    if verbose:
        random_idx = random.randint(0, len(completions)-1)
        prompts = kwargs.get('prompts')
        print(f"Prompt: {prompts[random_idx]}\n{'-'*50}\n{completions[random_idx]}'")

    return [1.0 if _match else 0.0 for _match in extracted_codes]

def finetune(config: TrainingConfig):
    mlflow.set_experiment(config.exp_name)

    grpo_config = GRPOConfig(
        output_dir=str(config.output_dir_grpo),
        logging_steps=10,
        max_steps=2000,
        learning_rate=1e-6,
        gradient_accumulation_steps=16,
        report_to="mlflow",
        bf16=True,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=3,
        max_prompt_length=256,
        max_completion_length=1024,
        per_device_train_batch_size=1,
        num_generations=8,
        temperature=0.8,
        use_vllm=True,
        vllm_server_host=os.environ.get("MASTER", "localhost"),
        vllm_server_port=int(os.environ.get("SERVER_PORT", 45678))
    )

    train_ds = GRPOCodeDataset(
        config=config,
        split="train"
    )
    val_ds = GRPOCodeDataset(
        config=config,
        split="test"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    template_rewards_fn_verbose = partial(template_rewards, verbose=config.verbose_grpo)
    template_rewards_fn_verbose.__name__ = "template_rewards"

    compile_rewards_fn_verbose = partial(compile_rewards, config=config)
    compile_rewards_fn_verbose.__name__ = "compile_rewards"

    trainer = GRPOTrainer(
        model=str(config.model_path),
        reward_funcs=[template_rewards_fn_verbose, compile_rewards_fn_verbose],
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    print(trainer.model)
    trainer.train()

