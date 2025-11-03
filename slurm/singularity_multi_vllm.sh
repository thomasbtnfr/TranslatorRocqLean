eval $(opam env)

export PYTHONPATH=/lustre/fswork/projects/rech/hir/uxp55sd/TranslatorRocqLean:$PYTHONPATH

trl vllm-serve \
    --model /lustre/fsmisc/dataset/HuggingFace_Models/bigcode/starcoder2-15b \
    --host $MASTER \
    --port $SERVER_PORT \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max_model_len 4096
