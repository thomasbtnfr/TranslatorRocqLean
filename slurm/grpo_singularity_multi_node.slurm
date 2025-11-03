#! /bin/bash

#SBATCH --job-name=GRPO
#SBATCH --account=wrx@h100
#SBATCH --constraint=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=slurm_grpo.out
#SBATCH --error=slurm_grpo.err
#SBATCH --exclusive
#SBATCH --qos=qos_gpu_h100-t3

cd ${SLURM_SUBMIT_DIR}

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export MASTER=${NODELIST[0]}
MASTER_IP=$(hostname --ip-address)

export SERVER_PORT=45678
export OPENAI_BASE_URL="http://$MASTER:$SERVER_PORT"

echo "MASTER node: $MASTER"

srun -N1 -w $MASTER singularity exec --nv \
--bind /lustre/fsmisc/dataset/HuggingFace_Models:/lustre/fsmisc/dataset/HuggingFace_Models/ \
--bind /linkhome/rech/genini01/uxp55sd/.cache:/linkhome/rech/genini01/uxp55sd/.cache \
--bind /lustre/fsn1/projects/rech/hir/uxp55sd:/lustre/fsn1/projects/rech/hir/uxp55sd/ \
--bind /lustre/fswork/projects/rech/hir/uxp55sd:/lustre/fswork/projects/rech/hir/uxp55sd/ \
--env XDG_CACHE_HOME=/linkhome/rech/genini01/uxp55sd/cache \
/lustre/fsn1/singularity/images/uxp55sd/trl-rl-deepspeed.sif \
bash slurm/singularity_multi_vllm.sh &

echo $OPENAI_BASE_URL
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

srun -N $(( SLURM_NNODES-1 )) --ntasks-per-node 1 -x $MASTER singularity exec --nv \
--bind /lustre/fsmisc/dataset/HuggingFace_Models:/lustre/fsmisc/dataset/HuggingFace_Models/ \
--bind /linkhome/rech/genini01/uxp55sd/.cache:/linkhome/rech/genini01/uxp55sd/.cache \
--bind /lustre/fsn1/projects/rech/hir/uxp55sd:/lustre/fsn1/projects/rech/hir/uxp55sd/ \
--bind /lustre/fswork/projects/rech/hir/uxp55sd:/lustre/fswork/projects/rech/hir/uxp55sd/ \
--env XDG_CACHE_HOME=/linkhome/rech/genini01/uxp55sd/cache \
/lustre/fsn1/singularity/images/uxp55sd/trl-rl-deepspeed.sif \
bash slurm/singularity_multi_trl.sh
