#!/bin/bash
# =============================================================================
# HiPerGator SLURM job — mood push vs audio quality sweep (strength x guidance)
# =============================================================================
# Submit with:  sbatch run_sweep_quality.sh [checkpoint_dir] [extra sweep_quality.py args...]
#   e.g.        sbatch run_sweep_quality.sh output/job_perchan
#               sbatch run_sweep_quality.sh output/job_X --guidance 3 7:0.7 --edit_strengths 0.4 0.5
#   (extra args come last, so they override the defaults below)
# Monitor with: squeue -u $USER
# Result:       $CKPT_DIR/sweep_quality.csv + listen_quality/  (+ summary in the .out log)
# =============================================================================

#SBATCH --job-name=mood-sweepq
#SBATCH --output=logs/sweepq_%j.out
#SBATCH --error=logs/sweepq_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=asherwheatle@ufl.edu
#SBATCH --mail-type=ALL

module purge
module load cuda/12.8.1

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/musicdiffusion"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

mkdir -p logs

CKPT_DIR="${1:-}"
if [ -z "$CKPT_DIR" ]; then
    CKPT_DIR="$(ls -dt output/job_*/ 2>/dev/null | head -1)"
    CKPT_DIR="${CKPT_DIR%/}"
fi
if [ -z "$CKPT_DIR" ] || [ ! -f "$CKPT_DIR/diffusion.pt" ]; then
    echo "[ERROR] No diffusion.pt found in CKPT_DIR='$CKPT_DIR'." >&2
    exit 1
fi
echo "[RUN] Sweeping checkpoint dir: $CKPT_DIR"

DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"
CLAP_CKPT="music_audioset_epoch_15_esc_90.14.pt"

echo "[RUN] Starting on $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python sweep_quality.py \
    --ckpt_dir "$CKPT_DIR" \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --clap_ckpt "$CLAP_CKPT" \
    --n_songs 30 --edit_strengths 0.8 0.9 --guidance 5 7 10 --inits invert invert_src \
    "${@:2}"

echo "[RUN] Done on $(date)"
