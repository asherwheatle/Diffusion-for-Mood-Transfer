#!/bin/bash
# =============================================================================
# HiPerGator SLURM job — conditioning-path probe (text vs audio vs shifted text)
# =============================================================================
# Submit with:  sbatch run_probe.sh [checkpoint_dir]
#   e.g.        sbatch run_probe.sh output/job_41279148
#   (no arg -> probes the newest output/job_* directory)
# Monitor with: squeue -u $USER
# Result:       $CKPT_DIR/probe_conditioning.csv  (+ verdict in the .out log)
# =============================================================================

#SBATCH --job-name=mood-probe
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err
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

# --- Checkpoint dir: first arg wins, else the newest output/job_* directory.
CKPT_DIR="${1:-}"
if [ -z "$CKPT_DIR" ]; then
    CKPT_DIR="$(ls -dt output/job_*/ 2>/dev/null | head -1)"
    CKPT_DIR="${CKPT_DIR%/}"          # strip trailing slash left by `ls -d`
fi
if [ -z "$CKPT_DIR" ] || [ ! -f "$CKPT_DIR/diffusion.pt" ]; then
    echo "[ERROR] No diffusion.pt found in CKPT_DIR='$CKPT_DIR'." >&2
    echo "        Pass a trained checkpoint dir explicitly:" >&2
    echo "        sbatch run_probe.sh output/job_XXXXXX" >&2
    exit 1
fi
echo "[RUN] Probing checkpoint dir: $CKPT_DIR"

DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"
CLAP_CKPT="music_audioset_epoch_15_esc_90.14.pt"

python -c "import laion_clap" 2>/dev/null || {
    echo "[SETUP] Installing laion-clap + torchvision..."
    uv pip install "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
    uv pip install laion-clap
}
if [ ! -f "$CLAP_CKPT" ]; then
    echo "[SETUP] Downloading CLAP music checkpoint..."
    wget -q "https://huggingface.co/lukewys/laion_clap/resolve/main/$CLAP_CKPT"
fi

echo "[RUN] Starting conditioning probe on $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python probe_conditioning.py \
    --ckpt_dir "$CKPT_DIR" \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --clap_ckpt "$CLAP_CKPT" \
    --n_songs 10 --n_ref 24 --cfg_scale 5.0 --edit_strength 0.6

echo "[RUN] Done on $(date)"
echo "[RUN] Result in: $CKPT_DIR/probe_conditioning.csv"
