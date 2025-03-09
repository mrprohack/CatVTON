import os
import io
import base64
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import torch
from huggingface_hub import snapshot_download, HfFolder
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model.pipeline import CatVTONPix2PixPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="CatVTON P2P API", 
    description="Virtual Try-On API using CatVTON Pix2Pix",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure this based on your deployment
)

# Add rate limiter error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global variables to store models
pipeline_p2p = None
MODEL_CONFIG = {
    "base_model": "timbrooks/instruct-pix2pix",
    "resume_model": "zhengchong/CatVTON-MaskFree",  # This is a gated model
    "version": "mix-48k-1024",
    "local_model_path": os.getenv("LOCAL_MODEL_PATH", "models/CatVTON-MaskFree")  # Allow local model path override
}

def setup_huggingface_auth():
    """Setup HuggingFace authentication using token from environment variable"""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        HfFolder.save_token(hf_token)
        logger.info("HuggingFace token configured")
    else:
        logger.warning("No HuggingFace token found in environment. Access to gated models may be restricted.")

def init_models():
    global pipeline_p2p
    
    if pipeline_p2p is not None:
        return
    
    try:
        logger.info("Initializing models...")
        
        # Try to load from local path first
        model_path = MODEL_CONFIG["local_model_path"]
        if not os.path.exists(model_path):
            try:
                # Setup HuggingFace authentication
                setup_huggingface_auth()
                
                # Try to download from HuggingFace
                logger.info(f"Downloading model from {MODEL_CONFIG['resume_model']}...")
                model_path = snapshot_download(
                    repo_id=MODEL_CONFIG["resume_model"],
                    local_dir=MODEL_CONFIG["local_model_path"]
                )
            except GatedRepoError as e:
                error_msg = (
                    f"Cannot access gated model {MODEL_CONFIG['resume_model']}. "
                    "Please provide a HuggingFace token with access or use a local model path. "
                    f"Original error: {str(e)}"
                )
                logger.error(error_msg)
                raise HTTPException(status_code=503, detail=error_msg)
            except RepositoryNotFoundError as e:
                error_msg = f"Model repository {MODEL_CONFIG['resume_model']} not found. Please check the model path."
                logger.error(error_msg)
                raise HTTPException(status_code=503, detail=error_msg)
            except Exception as e:
                error_msg = f"Error downloading model: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=503, detail=error_msg)
        
        # Initialize the pipeline
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing pipeline on {device}...")
        
        pipeline_p2p = CatVTONPix2PixPipeline(
            base_ckpt=MODEL_CONFIG["base_model"],
            attn_ckpt=model_path,
            attn_ckpt_version=MODEL_CONFIG["version"],
            weight_dtype=init_weight_dtype("bf16"),
            use_tf32=True,
            device=device
        )
        logger.info("Models initialized successfully")
        
    except Exception as e:
        error_msg = f"Error initializing models: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)

def validate_image(image: Image.Image, min_size: int = 64, max_size: int = 4096) -> bool:
    width, height = image.size
    if width < min_size or height < min_size:
        raise HTTPException(status_code=400, detail=f"Image dimensions must be at least {min_size}x{min_size} pixels")
    if width > max_size or height > max_size:
        raise HTTPException(status_code=400, detail=f"Image dimensions must not exceed {max_size}x{max_size} pixels")
    return True

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.on_event("startup")
async def startup_event():
    init_models()

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up resources
    global pipeline_p2p
    if pipeline_p2p is not None:
        del pipeline_p2p
        torch.cuda.empty_cache()

@app.get("/health")
async def health_check():
    if pipeline_p2p is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "model_loaded": True}

@app.get("/model-info")
async def model_info():
    return {
        "base_model": MODEL_CONFIG["base_model"],
        "resume_model": MODEL_CONFIG["resume_model"],
        "version": MODEL_CONFIG["version"],
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/try-on")
@limiter.limit("10/minute")  # Adjust rate limit as needed
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
    
    Parameters:
    - **person_image**: Image of the person
    - **cloth_image**: Image of the clothing item
    - **num_steps**: Number of inference steps (10-100)
    - **guidance_scale**: Guidance scale (0.0-7.5)
    - **seed**: Random seed (-1 for random)
    - **width**: Image width (64-2048)
    - **height**: Image height (64-2048)
    """
    try:
        # Validate parameters
        if not (10 <= num_steps <= 100):
            raise HTTPException(status_code=400, detail="num_steps must be between 10 and 100")
        if not (0.0 <= guidance_scale <= 7.5):
            raise HTTPException(status_code=400, detail="guidance_scale must be between 0.0 and 7.5")
        if not (64 <= width <= 2048) or not (64 <= height <= 2048):
            raise HTTPException(status_code=400, detail="Image dimensions must be between 64 and 2048 pixels")
            
        # Read and convert images
        try:
            person_img = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
            cloth_img = Image.open(io.BytesIO(await cloth_image.read())).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
            
        # Validate images
        validate_image(person_img)
        validate_image(cloth_img)
        
        # Process images
        person_img = resize_and_crop(person_img, (width, height))
        cloth_img = resize_and_padding(cloth_img, (width, height))
        
        # Set generator for reproducibility
        generator = None
        if seed != -1:
            generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed)
        
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
        
        return JSONResponse(
            content={
                "status": "success",
                "result": result_base64
            }
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api_p2p_server:app", host="0.0.0.0", port=8000, reload=True) 
