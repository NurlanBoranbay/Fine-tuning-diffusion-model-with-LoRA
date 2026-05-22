#!/bin/bash
# =============================================================================
# setup_env.sh - One-time environment setup for SDXL LoRA training
# Run this ONCE on the login node (access) to set up your conda environment
# =============================================================================

echo "============================================="
echo "  Miles Morales LoRA - Environment Setup"
echo "============================================="

# Load standard Python module instead of Anaconda
module load Python/3.11.5-GCCcore-13.2.0

# Create a dedicated python virtual environment for LoRA training
echo "[1/5] Creating python virtual environment 'lora_train_env'..."
python3 -m venv lora_train_env

# Activate the environment
echo "[2/5] Activating environment..."
source lora_train_env/bin/activate

# Install PyTorch with CUDA support (compatible with H100 GPUs)
echo "[3/5] Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face libraries for diffusion model training
echo "[4/5] Installing diffusers, transformers, accelerate, and training dependencies..."
pip install diffusers[torch] transformers accelerate datasets
pip install peft bitsandbytes safetensors
pip install Pillow numpy tqdm wandb
pip install xformers prodigyopt

# Install kohya-ss/sd-scripts for LoRA training (industry-standard tool)
echo "[5/5] Installing kohya-ss sd-scripts for LoRA training..."
cd $HOME
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
pip install -r requirements.txt
pip install -e .

echo ""
echo "============================================="
echo "  Environment setup complete!"
echo "  To use: source lora_train_env/bin/activate"
echo "============================================="
