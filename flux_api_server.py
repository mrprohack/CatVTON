import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker
from model.flux.pipeline_flux_tryon import FluxTryOnPipeline
from utils import resize_and_crop, resize_and_padding

app = FastAPI(title="FLUX Try-On API", description="Virtual Try-On API using FLUX model")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models
pipeline_flux = None
mask_processor = None
automasker = None

# Configuration
CONFIG = {
    "width": 768,
    "height": 1024,
    "base_model_path": "Models/FLUX.1-Fill-dev",
    "resume_path": "zhengchong/CatVTON"
}

def init_models():
    global pipeline_flux, mask_processor, automasker
    
    if pipeline_flux is not None:
        return
    
    print("Initializing models...")
    
    # Download and setup repo
    repo_path = snapshot_download(repo_id=CONFIG["resume_path"])
    
    # Initialize FLUX pipeline
    pipeline_flux = FluxTryOnPipeline.from_pretrained(CONFIG["base_model_path"])
    pipeline_flux.load_lora_weights(
        os.path.join(repo_path, "flux-lora"), 
        weight_name='pytorch_lora_weights.safetensors'
    )
    pipeline_flux.to("cuda", torch.bfloat16)
    
    # Initialize mask processor
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, 
        do_normalize=False, 
        do_binarize=True, 
        do_convert_grayscale=True
    )
    
    # Initialize automasker
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device='cuda'
    )

@app.on_event("startup")
async def startup_event():
    init_models()

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    cloth_type: str = Form("upper"),
    num_steps: Optional[int] = Form(50),
    guidance_scale: Optional[float] = Form(30.0),
    seed: Optional[int] = Form(42),
    width: Optional[int] = Form(768),
    height: Optional[int] = Form(1024),
    mask_image: Optional[UploadFile] = None
):
    """
    Virtual try-on endpoint using FLUX model.
    
    - **person_image**: Image of the person
    - **cloth_image**: Image of the clothing item
    - **cloth_type**: Type of clothing (upper/lower/overall)
    - **num_steps**: Number of inference steps
    - **guidance_scale**: Guidance scale for inference
    - **seed**: Random seed (-1 for random)
    - **width**: Image width
    - **height**: Image height
    - **mask_image**: Optional mask image
    """
    try:
        # Read and convert images
        person_img = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
        cloth_img = Image.open(io.BytesIO(await cloth_image.read())).convert("RGB")
        
        # Process images
        person_img = resize_and_crop(person_img, (width, height))
        cloth_img = resize_and_padding(cloth_img, (width, height))
        
        # Handle mask
        if mask_image:
            mask = Image.open(io.BytesIO(await mask_image.read())).convert("L")
            mask = resize_and_crop(mask, (width, height))
            # Ensure binary mask
            mask_array = np.array(mask)
            mask_array[mask_array > 0] = 255
            mask = Image.fromarray(mask_array)
        else:
            mask = automasker(person_img, cloth_type)['mask']
        
        mask = mask_processor.blur(mask, blur_factor=9)
        
        # Set generator for reproducibility
        generator = None
        if seed != -1:
            generator = torch.Generator(device='cuda').manual_seed(seed)
        
        # Run inference
        result_image = pipeline_flux(
            image=person_img,
            condition_image=cloth_img,
            mask_image=mask,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Convert result to base64
        result_base64 = image_to_base64(result_image)
        
        return {
            "status": "success",
            "result": result_base64
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("flux_api_server:app", host="0.0.0.0", port=8001, reload=True) 
