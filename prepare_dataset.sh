#!/bin/bash
# =============================================================================
# prepare_dataset.sh - Prepare dataset structure for kohya-ss LoRA training
# Run this on the login node AFTER uploading your dataset to the cluster
# =============================================================================

echo "============================================="
echo "  Preparing Dataset for LoRA Training"
echo "============================================="

# Set the project directory on the cluster
# IMPORTANT: Change this to your actual path on the cluster
PROJECT_DIR="$HOME/SpiderMan_LoRA_Project"
DATASET_DIR="${PROJECT_DIR}/img"

# kohya-ss expects the folder structure:
#   img/
#     <num_repeats>_<concept_name>/
#       image1.jpg + image1.txt
#       image2.jpg + image2.txt
#       ...
#
# Your existing structure already matches this pattern:
#   img/20_ITSV_MilesSuit/   (20 repeats per epoch for ITSV images)
#   img/20_ATSV_MilesSuit/   (20 repeats per epoch for ATSV images)

echo "Checking dataset structure..."

# Count images in each folder
ITSV_COUNT=$(ls -1 ${DATASET_DIR}/20_ITSV_MilesSuit/*.jpg 2>/dev/null | wc -l)
ATSV_COUNT=$(ls -1 ${DATASET_DIR}/20_ATSV_MilesSuit/*.jpg 2>/dev/null | wc -l)

echo "  ITSV (Into the Spider-Verse) images: ${ITSV_COUNT}"
echo "  ATSV (Across the Spider-Verse) images: ${ATSV_COUNT}"
echo "  Total images: $((ITSV_COUNT + ATSV_COUNT))"

# Verify every image has a matching caption
echo ""
echo "Verifying caption files..."
MISSING=0
for img in ${DATASET_DIR}/20_ITSV_MilesSuit/*.jpg ${DATASET_DIR}/20_ATSV_MilesSuit/*.jpg; do
    caption="${img%.jpg}.txt"
    if [ ! -f "$caption" ]; then
        echo "  WARNING: Missing caption for $(basename $img)"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 0 ]; then
    echo "  All images have matching captions."
else
    echo "  WARNING: ${MISSING} images are missing captions!"
fi

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p ${PROJECT_DIR}/output/model
mkdir -p ${PROJECT_DIR}/output/logs
mkdir -p ${PROJECT_DIR}/output/samples

# Download the base SDXL model if not already present
echo ""
echo "Checking for base model..."
MODEL_DIR="${PROJECT_DIR}/models"
mkdir -p ${MODEL_DIR}

if [ ! -d "${MODEL_DIR}/stable-diffusion-xl-base-1.0" ]; then
    echo "Downloading Stable Diffusion XL base model..."
    echo "This may take a while (~6.5 GB)..."
    
    # Load Python and activate environment
    module load Python/3.11.5-GCCcore-13.2.0
    source lora_train_env/bin/activate
    
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='stabilityai/stable-diffusion-xl-base-1.0',
    local_dir='${MODEL_DIR}/stable-diffusion-xl-base-1.0',
    ignore_patterns=['*.safetensors', '*.ckpt', '*.bin'],  # Download config only first
)
print('Base model config downloaded. Full weights will be loaded during training.')
"
    echo "  Base model downloaded to ${MODEL_DIR}/stable-diffusion-xl-base-1.0"
else
    echo "  Base model already exists."
fi

echo ""
echo "============================================="
echo "  Dataset preparation complete!"
echo "  Ready to submit training job."
echo "============================================="
