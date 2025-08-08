# TranslatorRocqLean

## Install
`pip install --editable .`

## CLI
- General
  - `python -m trl_llm --help`
- Training
  - On Jean Zay: `module load pytorch-gpu/py3/2.7.0`
  - `python -m trl_llm train --help`
- Gradio
  - On Jean Zay: `module load vllm/0.7.1`
  - `python -m trl_llm gradio --help`

## MLFlow
- `mlflow ui --backend-store-uri {path_to_your_mlflow_directory}`