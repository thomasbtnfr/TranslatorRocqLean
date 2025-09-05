import os
import subprocess
from collections import defaultdict

import pandas as pd

from trl_llm.eval.utils import load_visual_prompt, extract_step, check_rocq_code, check_lean_code, \
    extract_code
from trl_llm.train.config import TrainingConfig


def compile_rocq(config: TrainingConfig) -> None:
    visual_prompt_content = load_visual_prompt(config.visual_prompt_generated_path)
    print(f"Loaded {len(visual_prompt_content)} visual prompts")

    success_cnt_step = defaultdict(int)
    sample_counts = defaultdict(int)
    records = []

    # Prepare Lean environment
    cwd = os.getcwd()
    os.chdir(config.compile_lean_dir_path)
    result = subprocess.run(["lake", "build"], capture_output=True, text=True)
    print(result)

    for sample in visual_prompt_content:
        step = extract_step(sample[0])
        success_cnt_step.setdefault(step, 0)
        sample_counts[step] += 1
        sample_index = sample_counts[step]
        print(f"Processing step {step}, sample {sample_index}")

        prompt, generated = sample[1], sample[2]
        extracted_code = extract_code(generated)
        has_compiled = False
        language = None

        if config.template_lean_to_rocq in prompt:
            language = "rocq"
            has_compiled = extracted_code and check_rocq_code(extracted_code)

        elif config.template_rocq_to_lean in prompt:
            language = "lean"
            has_compiled = extracted_code and check_lean_code(extracted_code, config)

        else:
            print("\t\t\tUnknown template")

        if extracted_code:
            if has_compiled:
                success_cnt_step[step] += 1
            else:
                print("*" * 100, f"Failed to compile step={step}, sample={sample_index}, language={language}")
        else:
            print(f"\tCan't parse step={step}, sample={sample_index}")

        records.append({
            "step": step,
            "sample_index": sample_index,
            "original_prompt": prompt,
            "generated": generated,
            "extracted_code": extracted_code,
            "has_compiled": has_compiled,
            "language": language
        })

    # Reporting
    print(f"Samples per step: {dict(sample_counts)}")
    print(f"Successful compilations per step: {dict(success_cnt_step)}")

    os.chdir(cwd)
    df = pd.DataFrame(records)
    stats = df.groupby(["step", "language"])["has_compiled"].value_counts().unstack(fill_value=0)
    print(stats)

    df.to_csv("output.csv", index=False, encoding="utf-8")