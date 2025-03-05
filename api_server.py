import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

app = FastAPI(title="CatVTON API", description="Virtual Try-On API using CatVTON")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models
pipeline = None
mask_processor = None
automasker = None

def init_models():
    global pipeline, mask_processor, automasker
    
    if pipeline is not None:
        return
    
    print("Initializing models...")
    base_model_path = "booksforcharlie/stable-diffusion-inpainting"
    resume_path = "zhengchong/CatVTON"
    
    repo_path = snapshot_download(repo_id=resume_path)
    
    pipeline = CatVTONPipeline(
        base_ckpt=base_model_path,
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
    guidance_scale: Optional[float] = Form(2.5),
    seed: Optional[int] = Form(42),
    width: Optional[int] = Form(768),
    height: Optional[int] = Form(1024)
):
    """
    Virtual try-on endpoint.
    
    - **person_image**: Image of the person
    - **cloth_image**: Image of the clothing item
    - **cloth_type**: Type of clothing (upper/lower/overall)
    - **num_steps**: Number of inference steps
    - **guidance_scale**: Guidance scale for inference
    - **seed**: Random seed (-1 for random)
    - **width**: Image width
    - **height**: Image height
    """
    try:
        # Read and convert images
        person_img = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
        cloth_img = Image.open(io.BytesIO(await cloth_image.read())).convert("RGB")
        
        # Process images
        person_img = resize_and_crop(person_img, (width, height))
        cloth_img = resize_and_padding(cloth_img, (width, height))
        
        # Generate mask
        mask = automasker(person_img, cloth_type)['mask']
        mask = mask_processor.blur(mask, blur_factor=9)
        
        # Set generator for reproducibility
        generator = None
        if seed != -1:
            generator = torch.Generator(device='cuda').manual_seed(seed)
        
        # Run inference
        result_image = pipeline(
            image=person_img,
            condition_image=cloth_img,
            mask=mask,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )[0]
        
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
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 
