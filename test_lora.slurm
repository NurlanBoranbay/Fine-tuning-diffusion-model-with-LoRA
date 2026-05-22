#!/bin/bash
# =============================================================================
# test_lora.slurm - SLURM batch script to test/infer with the trained LoRA
# NU HPC Irgetas Cluster - Single Node, Single H100 GPU
# =============================================================================

#SBATCH --job-name=TestLoRA                # Job name
#SBATCH --partition=H100                   # GPU partition on Irgetas
#SBATCH --qos=nu-h100                      # QoS for general NU researchers on H100
#SBATCH --nodes=1                          # Single node
#SBATCH --ntasks=1                         # Single task
#SBATCH --cpus-per-task=4                  # 4 CPU threads
#SBATCH --gres=gpu:1                       # 1 x NVIDIA H100
#SBATCH --mem=32G                          # 32 GB system RAM
#SBATCH --time=0-0:30:00                   # 30 minutes max
#SBATCH --output=output/logs/infer_%j.out  # Stdout log
#SBATCH --error=output/logs/infer_%j.err   # Stderr log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nurlan.boranbay@nu.edu.kz    # CHANGE THIS

# Environment setup
module purge
module load Python/3.11.5-GCCcore-13.2.0
source lora_train_env/bin/activate

cd $HOME/SpiderMan_LoRA_Project

echo "============================================="
echo "  Testing Miles Morales LoRA"
echo "============================================="

# Run inference script
python3 generate_samples.py

echo "Done! Check output/samples/ for generated images."
