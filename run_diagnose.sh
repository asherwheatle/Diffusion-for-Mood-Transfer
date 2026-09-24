#!/bin/bash
# =============================================================================
# HiPerGator SLURM job — per-stage diagnosis (BigVGAN vs AE vs diffusion prior)
# =============================================================================
# Submit with:  sbatch run_diagnose.sh [checkpoint_dir] [probe_npz]
#   e.g.        sbatch run_diagnose.sh output/job_perchan \
#                   output/job_42891316_ext/valence_probe.npz
#   (no arg -> the newest output/job_* directory; probe defaults to the one
#    evaluate.py cached in the checkpoint dir, if any)
# Monitor with: squeue -u $USER
# Result:       $CKPT_DIR/diagnose_stages.csv  (+ summary in the .out log)
# =============================================================================

#SBATCH --job-name=mood-diagnose
#SBATCH --output=logs/diagnose_%j.out
#SBATCH --error=logs/diagnose_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
PROBE_ARGS=()
if [ -n "${2:-}" ]; then
    PROBE_ARGS+=(--probe_path "$2")
fi
echo "[RUN] Diagnosing checkpoint dir: $CKPT_DIR"

DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"
CLAP_CKPT="music_audioset_epoch_15_esc_90.14.pt"

echo "[RUN] Starting on $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python diagnose_stages.py \
    --ckpt_dir "$CKPT_DIR" \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --clap_ckpt "$CLAP_CKPT" \
    --n_songs 30 --edit_strengths 0.4 0.6 --cfg_scale 7.0 \
    "${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"}"

echo "[RUN] Done on $(date)"
