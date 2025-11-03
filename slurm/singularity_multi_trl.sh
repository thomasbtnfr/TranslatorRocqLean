eval $(opam env)

export PYTHONPATH=/lustre/fswork/projects/rech/hir/uxp55sd/TranslatorRocqLean:$PYTHONPATH

accelerate launch --config_file slurm/accelerate_config.yaml \
-m trl_llm grpo -c slurm/config_grpo.json --verbose-grpo