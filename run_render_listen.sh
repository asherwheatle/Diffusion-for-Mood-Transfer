#!/bin/bash
# =============================================================================
# HiPerGator SLURM job — render sweep edits to WAV for listening
# =============================================================================
# Submit with:  sbatch run_render_listen.sh [checkpoint_dir] [extra render_listen.py args...]
#   e.g.        sbatch run_render_listen.sh output/job_perchan
#               sbatch run_render_listen.sh output/job_X --gt both
#   (extra args come last, so they override the defaults below)
# Monitor with: squeue -u $USER
# Result:       $CKPT_DIR/listen/<song>_<gt>/*.wav + listen/INDEX.txt
# =============================================================================

#SBATCH --job-name=mood-listen
#SBATCH --output=logs/listen_%j.out
#SBATCH --error=logs/listen_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
echo "[RUN] Rendering checkpoint dir: $CKPT_DIR"

DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"
CLAP_CKPT="music_audioset_epoch_15_esc_90.14.pt"

echo "[RUN] Starting on $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python render_listen.py \
    --ckpt_dir "$CKPT_DIR" \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --clap_ckpt "$CLAP_CKPT" \
    --n_songs 30 --melody_scale 1.0 --edit_strengths 0.6 0.8 0.95 --cfg_scale 7.0 \
    "${@:2}"

echo "[RUN] Done on $(date)"
