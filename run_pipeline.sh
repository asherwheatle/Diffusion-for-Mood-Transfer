#!/bin/bash
# =============================================================================
# HiPerGator SLURM job script — musicdiffusion pipeline
# =============================================================================
# Submit with:  sbatch run_pipeline.sh
# Monitor with: squeue -u $USER
# =============================================================================

#SBATCH --job-name=mood-diffusion-full
#SBATCH --output=logs/mood_full_%j.out
#SBATCH --error=logs/mood_full_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
# Single-GPU training. Extra CPUs feed the pinned-memory DataLoader workers
# (num_workers in config.py) plus the pin_memory thread.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --mail-user=asherwheatle@ufl.edu
#SBATCH --mail-type=ALL

# Fail loudly: abort on any unhandled error, unset variable, or broken pipe
# so setup problems don't silently fall through to a half-run pipeline.
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Load system modules
# ---------------------------------------------------------------------------
module purge
module load cuda/12.8.1            # B200 GPUs require CUDA >= 12.8

# ---------------------------------------------------------------------------
# 2. Install uv (user-local, no root needed) if not already present
# ---------------------------------------------------------------------------
# uv installs to $HOME/.local/bin on Linux
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "[SETUP] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # re-source in case installer updated PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# 3. Create venv and sync all dependencies from pyproject.toml
#    uv will pull torch/torchaudio from the CUDA 12.8 index automatically.
#    Venv lives in $HOME/.venvs — always writable, no root needed.
# ---------------------------------------------------------------------------
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/musicdiffusion"
mkdir -p "$HOME/.venvs"

echo "[SETUP] Syncing dependencies with uv..."
uv sync

# ---------------------------------------------------------------------------
# 4. Activate the environment
# ---------------------------------------------------------------------------
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# ---------------------------------------------------------------------------
# 4b. Ensure CLAP (and its torchvision dep) are installed. The diffusion phase
#     builds a CLAP text encoder; laion_clap imports torchvision, which isn't
#     pulled by `uv sync`. Runs after the sync so it self-heals if sync prunes
#     it. (Mirrors run_eval.sh — keep the two in step.)
# ---------------------------------------------------------------------------
CLAP_CKPT="music_audioset_epoch_15_esc_90.14.pt"
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

# ---------------------------------------------------------------------------
# 5. Create output and log directories
# ---------------------------------------------------------------------------
mkdir -p logs output

# ---------------------------------------------------------------------------
# 6. Clone BigVGAN from HuggingFace if not already present
#    flock ensures only one concurrent job performs the clone — others wait.
# ---------------------------------------------------------------------------
(
  flock -x 200
  if [ ! -d "bigvgan_v2_44khz_128band_512x" ]; then
      echo "[SETUP] Cloning BigVGAN model from HuggingFace..."
      module load git-lfs
      git lfs install
      git clone https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x
  else
      echo "[SETUP] BigVGAN already present, skipping clone."
  fi
) 200>/tmp/bigvgan_clone.lock

# ---------------------------------------------------------------------------
# 7. Run the pipeline
#    Default: each job gets its own output directory keyed by SLURM job ID,
#    so concurrent runs never clobber each other's results.
#
#    Resume: pass an existing output dir as the first argument to reuse its
#    checkpoints instead of starting fresh, e.g.
#        sbatch run_pipeline.sh output/job_40998207
#    Training then picks up from that dir's *_ckpt.pt — a completed autoencoder
#    (epoch == ae_epochs) is skipped and the run goes straight to diffusion.
# ---------------------------------------------------------------------------
RESUME_DIR="${1:-}"            # ${1:-} keeps this safe under `set -u`
if [ -n "$RESUME_DIR" ]; then
    OUTPUT_DIR="$RESUME_DIR"
    echo "[RUN] Resuming into existing output dir: $OUTPUT_DIR"
else
    OUTPUT_DIR="output/job_${SLURM_JOB_ID}"
fi
mkdir -p "$OUTPUT_DIR"

echo "[RUN] Starting pipeline on $(date)"
echo "[RUN] Job ID: $SLURM_JOB_ID  ->  output dir: $OUTPUT_DIR"
echo "[RUN] GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DATA_ROOT="/orange/ufdatastudios/asherwheatle/DEAM_audio"

# Run training under its own error guard so we can report the REAL outcome.
# Without this, a killed/crashed run still fell through to "[RUN] Done"
# and the job showed up as COMPLETED even with no saved model.
set +e
python mood_diffusion.py \
    --mode full \
    --audio_dir "$DATA_ROOT/MEMD_audio" \
    --annotations_dir "$DATA_ROOT/DEAM_Annotations" \
    --output_dir "$OUTPUT_DIR"
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "[RUN] Done on $(date)"
    echo "[RUN] Results saved to: $OUTPUT_DIR"
else
    echo "[RUN] FAILED (python exit $status) on $(date)" >&2
    echo "[RUN] Partial checkpoints (if any) in: $OUTPUT_DIR" >&2
fi
# Propagate the real exit code so SLURM records FAILED, not COMPLETED.
exit "$status"
