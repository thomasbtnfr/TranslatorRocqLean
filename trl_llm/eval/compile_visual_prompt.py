import os
import subprocess
from collections import defaultdict

import pandas as pd

from trl_llm.eval.utils import load_visual_prompt, extract_step, check_rocq_code, check_lean_code, \
    extract_code
from trl_llm.train.config import TrainingConfig


def compile_rocq(
        config: TrainingConfig
):
    visual_prompt_content = load_visual_prompt(config.visual_prompt_generated_path)
    print(f"{len(visual_prompt_content)=}")

    success_cnt_step = defaultdict(int)
    sample_counts = defaultdict(int)

    records = []

    os.chdir(config.compile_lean_dir_path)
    result = subprocess.run(["lake", "build"],
                            capture_output=True, text=True)
    print(result)

    for sample in visual_prompt_content:
        step = extract_step(sample[0])
        success_cnt_step.setdefault(step, 0)

        sample_counts[step] += 1
        print(f"Processing step {step}, sample {sample_counts[step]}")

        if config.template_lean_to_rocq in sample[1]:
            has_compiled = False

            if rocq_code := extract_code(sample[2]):
                if has_compiled := check_rocq_code(rocq_code):
                    success_cnt_step[step] += 1
                else:
                    print("*" * 100, f"Failed to compile {step=}, {sample_counts[step]}")
            else:
                print(f"\tCan't parse {step=}, {sample_counts[step]}")

            records.append({
                "step": step,
                "sample_index": sample_counts[step],
                "original_prompt": sample[1],
                "generated": sample[2],
                "extracted_code": rocq_code,
                "has_compiled": has_compiled
            })

        elif config.template_rocq_to_lean in sample[1]:
            has_compiled = False

            if lean_code := extract_code(sample[2]):
                if has_compiled := check_lean_code(lean_code, config):
                    success_cnt_step[step] += 1
                else:
                    print("*" * 100, f"Failed to compile {step=}, {sample_counts[step]}")
            else:
                print(f"\tCan't parse {step=}, {sample_counts[step]}")

            records.append({
                "step": step,
                "sample_index": sample_counts[step],
                "original_prompt": sample[1],
                "generated": sample[2],
                "extracted_code": lean_code,
                "has_compiled": has_compiled
            })
        else:
            print("\t\t\tUnknown template!")

    print(f"Samples per step: {dict(sample_counts)}")
    print(f"Successful compilations per step: {dict(success_cnt_step)}")

    df = pd.DataFrame(records)
    df.to_csv("output.csv", index=False, encoding="utf-8")