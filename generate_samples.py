"""
generate_samples.py - Generate sample images using the trained Miles Morales LoRA

This script loads the base SDXL model, applies the trained LoRA weights,
and generates a batch of test images with various prompts.

Usage:
    Submit via test_lora.slurm on the HPC cluster, or run locally:
    python3 generate_samples.py
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from pathlib import Path
import os

def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Path to the trained LoRA weights
    # Adjust epoch number as needed (saved every 2 epochs, training runs 16 epochs)
    LORA_PATH = "./output/model/miles_morales_spiderverse_lora.safetensors"
    
    OUTPUT_DIR = "./output/samples"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # LoRA strength (0.0 = no effect, 1.0 = full effect)
    LORA_SCALE = 0.8
    
    # Test prompts covering both suit styles
    PROMPTS = [
        # ITSV (Into the Spider-Verse) - spraypainted suit style
        "ITSV_MilesSuit, spiderverse style, spraypainted suit, "
        "dynamic action shot of the character swinging through a neon-lit city at night, "
        "halftone dots, comic book style, vibrant colors, masterpiece, best quality",
        
        # ATSV (Across the Spider-Verse) - sleek suit style
        "ATSV_MilesSuit, spiderverse style, sleek suit, "
        "portrait shot looking over shoulder with city skyline behind, "
        "chromatic aberration, glowing effects, comic style, masterpiece, best quality",
        
        # ITSV close-up
        "ITSV_MilesSuit, spiderverse style, spraypainted suit, "
        "close-up of mask face with detailed crosshatch patterns, "
        "dramatic red and blue lighting, halftone background, masterpiece",
        
        # ATSV action scene
        "ATSV_MilesSuit, spiderverse style, sleek suit, "
        "full body shot leaping between dimensions, pink and purple portal energy, "
        "reality glitch effects, dynamic pose, masterpiece, best quality",
        
        # Mixed - creative prompt
        "ITSV_MilesSuit, spiderverse style, spraypainted suit, "
        "standing on the edge of a skyscraper looking down at the city below, "
        "sunset golden hour lighting, wind blowing, comic panel borders, masterpiece",
    ]
    
    NEGATIVE_PROMPT = (
        "low quality, worst quality, blurry, deformed, disfigured, "
        "bad anatomy, bad hands, missing fingers, extra digits, "
        "watermark, text, signature, realistic, photorealistic"
    )
    
    # =========================================================================
    # Load Model
    # =========================================================================
    print("Loading SDXL base model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    
    # Use Euler Ancestral scheduler for good quality
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA weights
    if Path(LORA_PATH).exists():
        print(f"Loading LoRA weights from: {LORA_PATH}")
        pipe.load_lora_weights(LORA_PATH)
    else:
        print(f"WARNING: LoRA weights not found at {LORA_PATH}")
        print("Generating with base model only (for testing pipeline).")
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Enable memory optimizations
    pipe.enable_xformers_memory_efficient_attention()
    
    # =========================================================================
    # Generate Images
    # =========================================================================
    print(f"\nGenerating {len(PROMPTS)} images...")
    
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] Generating: {prompt[:80]}...")
        
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            guidance_scale=7.0,
            width=1024,
            height=1024,
            cross_attention_kwargs={"scale": LORA_SCALE},
            generator=torch.Generator(device="cuda").manual_seed(42 + i),
        ).images[0]
        
        # Save image
        filename = f"sample_{i+1:02d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        print(f"  Saved: {filepath}")
    
    print(f"\n{'='*50}")
    print(f"  All {len(PROMPTS)} images generated!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
