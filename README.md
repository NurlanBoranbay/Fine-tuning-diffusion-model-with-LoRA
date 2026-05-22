# Fine-Tuning Stable Diffusion XL (SDXL) with LoRA: Miles Morales Spider-Verse

This repository contains the training scripts, configuration, and sample prompts used to fine-tune Stable Diffusion XL (SDXL) on the distinctive artistic styles of Miles Morales from *Into the Spider-Verse* (ITSV) and *Across the Spider-Verse* (ATSV).

The training was performed on the **Nazarbayev University (NU) High-Performance Computing System (HPCS) Irgetas Cluster** using **4x NVIDIA H100 (80GB) GPUs** (with options for 1-GPU and 2-GPU scaling).

---

## 🎨 Model Details & Trigger Words

This LoRA captures two specific suit styles along with the general chromatic aberration, halftone dots, and comic book aesthetic of the films:

*   **LoRA Weight/Strength**: `0.8` (recommended)
*   **Resolution**: `1024 x 1024` pixels (standard SDXL resolution)
*   **Trigger Words**:
    *   `spiderverse style` (activates the general aesthetic, halftone textures, and comic book shading)
    *   `ITSV_MilesSuit, spraypainted suit` (activates the Into the Spider-Verse spraypainted suit style)
    *   `ATSV_MilesSuit, sleek suit` (activates the Across the Spider-Verse sleek, redesigned suit style)

---

## 📁 Repository Structure

*   [train_lora.slurm](file:///c:/Users/A/Downloads/SpiderMan_LoRA_Project/train_lora.slurm) - SLURM batch script for multi-GPU training on the NU HPCS cluster.
*   [training_config.toml](file:///c:/Users/A/Downloads/SpiderMan_LoRA_Project/training_config.toml) - Alternative `kohya-ss` training config.
*   [generate_samples.py](file:///c:/Users/A/Downloads/SpiderMan_LoRA_Project/generate_samples.py) - Standalone python script to run local inference using `diffusers`.
*   [sample_prompts.txt](file:///c:/Users/A/Downloads/SpiderMan_LoRA_Project/sample_prompts.txt) - List of evaluation prompts used during training.
*   [prepare_dataset.sh](file:///c:/Users/A/Downloads/SpiderMan_LoRA_Project/prepare_dataset.sh) - Helper script to prepare images and captions.
*   [setup_env.sh](file:///c:/Users/A/Downloads/SpiderMan_LoRA_Project/setup_env.sh) - Script to set up the virtual python environment.
*   `img/` - Dataset folder containing images and corresponding caption text files.

---

## 🚀 Training on NU HPCS (SLURM)

To run the training on the Nazarbayev University HPC cluster:

1.  **Set up the environment** (one-time setup on a login node):
    ```bash
    bash setup_env.sh
    ```
2.  **Submit the SLURM job**:
    ```bash
    sbatch train_lora.slurm
    ```

> [!NOTE]
> The current [train_lora.slurm](file:///c:/Users/A/Downloads/SpiderMan_LoRA_Project/train_lora.slurm) script is pre-configured for **2x NVIDIA H100 GPUs** with an effective batch size of `16` (`2 GPUs * batch_size 2 * gradient_accumulation 4`). If you wish to use 4 GPUs, scale the resources and adjust `--num_processes=4` and `--gradient_accumulation_steps=2`.

---

## 💻 Running Inference in Web UI (e.g., Stable Diffusion WebUI Forge / AUTOMATIC1111)

1.  **Move the LoRA file**: Put the trained `miles_morales_spiderverse_lora.safetensors` file inside the `webui/models/Lora/` directory.
2.  **Select Base Checkpoint**: Choose `sd_xl_base_1.0.safetensors` in the top-left checkpoint dropdown.
3.  **Set Resolution**: Change Width and Height to **`1024 x 1024`**.
4.  **Use prompt triggers**: Include the trigger words and the LoRA syntax in your prompt:
    ```text
    ITSV_MilesSuit, spiderverse style, spraypainted suit, dynamic action shot of a superhero swinging between neon city buildings at night, comic book style, halftone patterns <lora:miles_morales_spiderverse_lora:0.8>
    ```

---

## ⚙️ Recommended Git Workflow for this Repository

### What to Commit:
*   ✅ All configuration scripts (`.slurm`, `.sh`, `.toml`, `.py`, `.txt`).
*   ✅ The dataset images and captions (`img/` directory).
*   ✅ Evaluation/sample output images (`output/samples/`) showing model performance.

### What NOT to Commit:
*   ❌ **The Base SDXL Model**: (Stability AI's model is 6.9 GB and already hosted on Hugging Face).
*   ❌ **The LoRA `.safetensors` file**: At **170.5 MB**, this file exceeds GitHub's **100 MB hard file size limit** and will block your push. 

### Best Practice for Sharing the LoRA File:
Upload the `miles_morales_spiderverse_lora.safetensors` model file to **[Hugging Face Models](https://huggingface.co/)** or **[Civitai](https://civitai.com/)** and link it below:

*   🔗 **Download Trained LoRA weights:** `[Insert Hugging Face/Civitai Link Here]`
