#!/bin/bash
# =============================================================================
# train_lora.slurm - SLURM batch script for Miles Morales LoRA training
# NU HPC Irgetas Cluster - Single Node, 1x H100 GPU (Single-GPU)
# =============================================================================
# Student QoS (stud) limits on Irgetas H100 partition:
#   Max jobs: 4 | Max cores: 192 | Max threads: 384 | Max GPUs: 4
# This script uses 1 GPU allocation for faster queue time and identical quality.
# =============================================================================

# ---------------------------------------------------
# SLURM Resource Directives
# ---------------------------------------------------
#SBATCH --job-name=MilesLoRA              # Job name shown in squeue
#SBATCH --partition=H100                  # GPU partition on Irgetas cluster
#SBATCH --qos=nu-h100                     # QoS for general NU researchers on H100
#SBATCH --nodes=1                         # Single node (each node has 4x H100)
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=32                # 32 CPU threads for data loading (2 GPUs × 16)
#SBATCH --gres=gpu:2                      # Request 2 x NVIDIA H100 GPUs (80GB each)
#SBATCH --mem=128G                        # 128 GB system RAM for multi-GPU allocation
#SBATCH --time=0-12:00:00                 # Max runtime: 12 hours
#SBATCH --output=output/logs/slurm_%j.out # SLURM stdout log
#SBATCH --error=output/logs/slurm_%j.err  # SLURM stderr log
#SBATCH --mail-type=BEGIN,END,FAIL        # Email on job start, end, or failure
#SBATCH --mail-user=Nurlan.boranbay@nu.edu.kz   # CHANGE THIS to your NU email

# ---------------------------------------------------
# Environment Setup
# ---------------------------------------------------
echo "============================================="
echo "  Miles Morales LoRA Training Job"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "  GPUs:   2x NVIDIA H100 (Multi-GPU)"
echo "  Date:   $(date)"
echo "============================================="

# Load required modules
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate our training environment
source lora_train_env/bin/activate

# Set the working directory to the project folder
# IMPORTANT: Change this to match your actual path on the cluster
cd $HOME/SpiderMan_LoRA_Project

# Create output directories if they don't exist
mkdir -p output/model output/logs output/samples

# ---------------------------------------------------
# GPU & CUDA Diagnostics
# ---------------------------------------------------
echo ""
echo "--- GPU Information (2x H100) ---"
nvidia-smi
echo ""
echo "--- CUDA & GPU Check ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB')
"
echo ""

# ---------------------------------------------------
# Multi-GPU Training Execution
# ---------------------------------------------------
echo "--- Starting Multi-GPU LoRA Training (2x H100) ---"
echo "Start time: $(date)"
echo ""

# Use accelerate to launch kohya-ss sdxl_train_network.py across 2 GPUs
# With 2 GPUs: processes batch_size=2, grad_accum=4, total effective batch = 2 × 2 GPUs × 4 grad_accum = 16 (identical to original training setup)
accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision="bf16" \
    $HOME/sd-scripts/sdxl_train_network.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir="./img" \
    --output_dir="./output/model" \
    --output_name="miles_morales_spiderverse_lora" \
    --save_model_as="safetensors" \
    --save_precision="fp16" \
    --save_every_n_epochs=2 \
    --resolution="1024,1024" \
    --train_batch_size=2 \
    --max_train_epochs=32 \
    --seed=42 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=4 \
    --mixed_precision="bf16" \
    --min_snr_gamma=5 \
    --enable_bucket \
    --min_bucket_reso=512 \
    --max_bucket_reso=2048 \
    --bucket_reso_steps=64 \
    --caption_extension=".txt" \
    --keep_tokens=1 \
    --cache_latents \
    --cache_latents_to_disk \
    --cache_text_encoder_outputs \
    --cache_text_encoder_outputs_to_disk \
    --optimizer_type="Prodigy" \
    --learning_rate=1.0 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=100 \
    --optimizer_args "weight_decay=0.01" "decouple=True" "use_bias_correction=True" "d_coef=2.0" \
    --network_module="networks.lora" \
    --network_dim=32 \
    --network_alpha=16 \
    --network_train_unet_only \
    --max_token_length=225 \
    --noise_offset=0.0357 \
    --adaptive_noise_scale=0.00357 \
    --multires_noise_iterations=6 \
    --multires_noise_discount=0.3 \
    --max_data_loader_n_workers=8 \
    --persistent_data_loader_workers \
    --sample_every_n_epochs=4 \
    --sample_sampler="euler_a" \
    --sample_prompts="./sample_prompts.txt" \
    --logging_dir="./output/logs" \
    --log_prefix="miles_morales" \
    --sdpa

# ---------------------------------------------------
# Post-Training Summary
# ---------------------------------------------------
echo ""
echo "--- Training Complete ---"
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh output/model/
echo ""
echo "Sample images (if generated):"
ls -lh output/model/sample/ 2>/dev/null || echo "  No samples directory found."
echo ""
echo "============================================="
echo "  Job ${SLURM_JOB_ID} finished successfully!"
echo "============================================="
