#!/bin/bash
#SBATCH --job-name=fast_sdp
#SBATCH -A llc@v100                     # Compte GPU
#SBATCH --partition=gpu_p13            # Partition GPU
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=40             # CPUs dispo
#SBATCH --time=4:00:00                # Temps max
#SBATCH --output=logs/slurm-%x-%j.out  # Log SLURM standard
#SBATCH --error=logs/slurm-%x-%j.err   # Log SLURM erreur

# Charge les modules nécessaires
module purge
source ~/.bashrc
module load pytorch-gpu/py3/2.0.0
module unload openmpi
module load openmpi/4.1.1

# Active ton venv
source /lustre/fswork/projects/rech/llc/uvq13au/venvs/fastsdp/bin/activate

# Crée le répertoire de logs si besoin
DATE=$(date +%Y_%m_%d_%Hh%M_%Ss)
LOG_DIR="results/benchmark/${NETWORK_NAME}-${EPSILON}/${DATE}_${NAME_RUN}"
mkdir -p "$LOG_DIR"

# Redirige stdout et stderr
exec > >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.err" >&2)

echo "Network: $NETWORK_NAME"
echo "Run name: $NAME_RUN"
echo "Job ID: $SLURM_JOB_ID"

# Appel du script métier avec les arguments du SLURM
python -c "import torch; print(torch.cuda.is_available())"

echo "Network: $NETWORK_NAME"
echo "Run name: $NAME_RUN"

python src/certification_problem.py "$NETWORK_NAME" "$NAME_RUN"
#sbatch --export=NETWORK_NAME=$NETWORK_NAME,NAME_RUN=$NAME_RUN scripts/run_fastsdp_job.slurm