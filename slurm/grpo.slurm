#! /bin/bash

#SBATCH --job-name=GRPO
#SBATCH --account=wrx@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_grpo.out
#SBATCH --error=slurm_grpo.err
#SBATCH --exclusive
#SBATCH --qos=qos_gpu_h100-dev

cd ${SLURM_SUBMIT_DIR}

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export MASTER=${NODELIST[0]}
MASTER_IP=$(hostname --ip-address)

export SERVER_PORT=45678
export OPENAI_BASE_URL="http://$MASTER:$SERVER_PORT"

srun -N1 -w $MASTER trl vllm-serve \
    --model /lustre/fsmisc/dataset/HuggingFace_Models/bigcode/starcoder2-3b \
    --host $MASTER \
    --port $SERVER_PORT \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max_model_len 4096 &

CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model /lustre/fsmisc/dataset/HuggingFace_Models/bigcode/starcoder2-3b \
    --host $MASTER \
    --port $SERVER_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max_model_len 4096 &

unset HTTP_PROXY
unset http_proxy

ATTEMPT=0
DELAY=5
MAX_ATTEMPTS=60
until curl -s -o /dev/null -w "%{http_code}" $OPENAI_BASE_URL | grep -E "^4[0-9]{2}$"; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "$ATTEMPT attempts"
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        echo "Failed: the server did not respond any of the $MAX_ATTEMPTS requests."
        exit 1
    fi
    echo "vllm serve is not ready yet"
    sleep $DELAY
done

module purge
conda deactivate
module load arch/h100
module load pytorch-gpu/py3/2.8.0
export PYTHONUSERBASE=/lustre/fswork/projects/rech/hir/uxp55sd/.local_trl

srun -N $(( SLURM_NNODES-1 )) --ntasks-per-node 1 -x $MASTER idr_accelerate \
    --config_file slurm/accelerate_config.yaml -m trl_llm grpo -c slurm/config_grpo.json

# CUDA_VISIBLE_DEVICES=1 idr_accelerate \
#      --config_file accelerate_config.yaml -m trl_llm grpo -c config_grpo.json

CUDA_VISIBLE_DEVICES=2,3 accelerate launch -m trl_llm grpo -c /lustre/fswork/projects/rech/hir/uxp55sd/TranslatorRocqLean/slurm/config_grpo.json

# CUDA_VISIBLE_DEVICES=2 python -m trl_llm grpo -c /lustre/fswork/projects/rech/hir/uxp55sd/TranslatorRocqLean/slurm/config_grpo.json