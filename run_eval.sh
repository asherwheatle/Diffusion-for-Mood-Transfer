#!/bin/bash
# =============================================================================
# HiPerGator SLURM job — CLAP + chroma evaluation of the mood-editing model
# =============================================================================
# Submit with:  sbatch run_eval.sh [checkpoint_dir] [cfg_scale] [edit_strength]
#   e.g.        sbatch run_eval.sh output/job_40998207
#               sbatch run_eval.sh output/job_40998207 7.0 0.8
#   (no arg -> evaluates the newest output/job_* directory)
#   (defaults: cfg_scale 5.0, edit_strength 0.6)
# Monitor with: squeue -u $USER
# Results land in $CKPT_DIR: eval_edits.csv, clap_validation.csv, eval_summary.txt
# =============================================================================

#SBATCH --job-name=mood-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
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

# --- Checkpoint dir: pass one as the first argument, otherwise default to the
#     newest output/job_* directory.  e.g.  sbatch run_eval.sh output/job_40998207
CKPT_DIR="${1:-}"
if [ -z "$CKPT_DIR" ]; then
    CKPT_DIR="$(ls -dt output/job_*/ 2>/dev/null | head -1)"
    CKPT_DIR="${CKPT_DIR%/}"          # strip trailing slash left by `ls -d`
fi
if [ -z "$CKPT_DIR" ] || [ ! -f "$CKPT_DIR/diffusion.pt" ]; then
    echo "[ERROR] No diffusion.pt found in CKPT_DIR='$CKPT_DIR'." >&2
    echo "        Pass a trained checkpoint dir explicitly:" >&2
    echo "        sbatch run_eval.sh output/job_XXXXXX" >&2
    exit 1
fi
echo "[RUN] Evaluating checkpoint dir: $CKPT_DIR"

# --- Data / CLAP paths (edit if these move) ---
DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"
CLAP_CKPT="music_audioset_epoch_15_esc_90.14.pt"

# --- Make sure CLAP (and its torchvision dep) are installed ---
python -c "import laion_clap" 2>/dev/null || {
    echo "[SETUP] Installing laion-clap + torchvision..."
    # torchvision 0.22.0 matches torch 2.7.0/cu128; laion_clap imports it
    uv pip install "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
    uv pip install laion-clap
}
if [ ! -f "$CLAP_CKPT" ]; then
    echo "[SETUP] Downloading CLAP music checkpoint..."
    wget -q "https://huggingface.co/lukewys/laion_clap/resolve/main/$CLAP_CKPT"
fi

echo "[RUN] Starting evaluation on $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Guidance settings. These default to the values the conditioning diagnostics
# use (probe_conditioning.py / ablate_melody.py), NOT to config.py's cfg_scale
# of 1.5 — that is a training-time default, and evaluating at it measured the
# model at a third of the guidance the probes showed conditioning needs.
# Override per run:  sbatch run_eval.sh output/job_XXXXXX 7.0 0.8
CFG_SCALE="${2:-5.0}"
EDIT_STRENGTH="${3:-0.6}"
echo "[RUN] cfg_scale=$CFG_SCALE  edit_strength=$EDIT_STRENGTH"

python evaluate.py \
    --ckpt_dir "$CKPT_DIR" \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --clap_ckpt "$CLAP_CKPT" \
    --n_songs 20 --n_val 100 \
    --cfg_scale "$CFG_SCALE" --edit_strength "$EDIT_STRENGTH"

echo "[RUN] Done on $(date)"
echo "[RUN] Results in: $CKPT_DIR"
