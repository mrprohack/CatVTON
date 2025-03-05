import argparse
import os
from PIL import Image
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def ensure_extension(file_path, default_ext='.png'):
    """Ensure the file path has an image extension"""
    _, ext = os.path.splitext(file_path)
    if not ext:
        file_path += default_ext
    return file_path

def parse_args():
    parser = argparse.ArgumentParser(description="Run CatVTON inference from command line")
    parser.add_argument(
        "--person_image",
        type=str,
        required=True,
        help="Path to the person image"
    )
    parser.add_argument(
        "--cloth_image",
        type=str,
        required=True,
        help="Path to the cloth image"
    )
    parser.add_argument(
        "--cloth_type",
        type=str,
        default="upper",
        choices=["upper", "lower", "overall"],
        help="Type of clothing to try on"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save the output image"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="Guidance scale for inference"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (-1 for random)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",
        help="Path to base model"
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help="Path to CatVTON checkpoint"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output path has extension and directory exists
    args.output_path = ensure_extension(args.output_path)
    ensure_directory_exists(args.output_path)
    
    # Download and initialize model
    print("Initializing models...")
    repo_path = snapshot_download(repo_id=args.resume_path)
    
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_model_path,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("bf16"),
        use_tf32=True,
        device='cuda'
    )
    
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, 
        do_normalize=False, 
        do_binarize=True, 
        do_convert_grayscale=True
    )
    
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device='cuda'
    )

    # Load and process images
    print("Processing images...")
    person_image = Image.open(args.person_image).convert("RGB")
    cloth_image = Image.open(args.cloth_image).convert("RGB")
    
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Generate mask
    print("Generating mask...")
    mask = automasker(person_image, args.cloth_type)['mask']
    mask = mask_processor.blur(mask, blur_factor=9)
    
    # Set generator for reproducibility
    generator = None
    if args.seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(args.seed)
    
    # Run inference
    print("Running inference...")
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    )[0]
    
    # Save result
    print(f"Saving result to {args.output_path}")
    result_image.save(args.output_path)
    print("Done!")

if __name__ == "__main__":
    main() 
