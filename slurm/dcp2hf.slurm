#! /bin/bash

#SBATCH --job-name=TRL
#SBATCH --account=xyi@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00             # 2 hours is more than sufficient for MiniF2F, Putnam and dcp2hf (not for doc)
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --exclusive

cd ${SLURM_SUBMIT_DIR}

module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0

export PYTHONUSERBASE=/lustre/fswork/projects/rech/hir/uxp55sd/.local_trl

set -x

srun python -m trl_llm dcp2hf -c slurm/config.json \
    --model_name starcoder2-15b \
    --resume_from_step 8000 \
    --exp-name starcoder2-finetuning-doc \
    --run-name starcoder2-finetuning-doc-1757493881836434641 \
    --dcp-to-hf-saved-name starcoder2-instruct-doc-8000

