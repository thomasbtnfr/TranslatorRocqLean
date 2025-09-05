# TranslatorRocqLean

## Install
```bash
git clone https://github.com/thomasbtnfr/TranslatorRocqLean.git
cd TranslatorRocqLean
pip install --editable .
```

## CLI
- General
  - `python -m trl_llm --help`
- Training
  - On Jean Zay: `module load pytorch-gpu/py3/2.7.0`
  - `python -m trl_llm train --help`
  - See `slurm/train.slurm` and `slurm/train_specialization.slurm`
- Gradio
  - On Jean Zay: `module load vllm/0.7.1`
  - `python -m trl_llm gradio --help`
- Checkpoint DCP (Distributed Checkpoint) to HF (HuggingFace) format
  - `python -m trl_llm dcp2hf --help`
  - See `slurm/dcp2hf.slurm`
- Compile Rocq/Lean samples from visual prompt files
  - `python -m trl_llm compile_visual_prompt --help`
  - No GPU is required. Rocq and Lean must be installed. Some tips:
    - Rocq:
      - Attention to the version of Rocq installed. There may be incompatibilities/conflicts. For PutnamBench, follow these [instructions](https://github.com/trishullab/PutnamBench/blob/main/coq/setup.sh).
    - Lean:
      - After [installing Lean](https://leanprover-community.github.io/install/linux.html), you need to create a project that includes Mathlib. To do this, follow these [instructions](https://leanprover-community.github.io/install/project.html#creating-a-lean-project).
  - One the environments have been created, the following command should work:
    - ```bash
      python -m trl_llm compile_visual_prompt
      --visual_prompt_generated_path visual_prompt.json
      --compile-lean-dir-path TranslatorRocqLean/CompileLean/compile_lean/ 
      --compile-lean-file-path CompileLean.lean```
      
## MLFlow
- `mlflow ui --backend-store-uri {path_to_your_mlflow_directory}`