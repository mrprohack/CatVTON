import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import torch
from huggingface_hub import snapshot_download

from model.pipeline import CatVTONPix2PixPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

app = FastAPI(title="CatVTON P2P API", description="Virtual Try-On API using CatVTON Pix2Pix")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models
pipeline_p2p = None

def init_models():
    global pipeline_p2p
    
    if pipeline_p2p is not None:
        return
    
    print("Initializing models...")
    p2p_base_model_path = "timbrooks/instruct-pix2pix"
    p2p_resume_path = "zhengchong/CatVTON-MaskFree"
    
    repo_path = snapshot_download(repo_id=p2p_resume_path)
    
    pipeline_p2p = CatVTONPix2PixPipeline(
        base_ckpt=p2p_base_model_path,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix-48k-1024",
        weight_dtype=init_weight_dtype("bf16"),
        use_tf32=True,
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
async def try_on_p2p(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    num_steps: Optional[int] = Form(50),
    guidance_scale: Optional[float] = Form(2.5),
    seed: Optional[int] = Form(42),
    width: Optional[int] = Form(768),
    height: Optional[int] = Form(1024)
):
    """
    Virtual try-on endpoint using Pix2Pix pipeline (mask-free).
    
    - **person_image**: Image of the person
    - **cloth_image**: Image of the clothing item
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
        
        # Set generator for reproducibility
        generator = None
        if seed != -1:
            generator = torch.Generator(device='cuda').manual_seed(seed)
        
        # Run inference
        result_image = pipeline_p2p(
            image=person_img,
            condition_image=cloth_img,
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
    uvicorn.run("api_p2p_server:app", host="0.0.0.0", port=8000, reload=True) 
