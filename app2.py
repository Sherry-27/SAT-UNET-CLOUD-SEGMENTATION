from contextlib import asynccontextmanager
from fastapi import FastAPI, File, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image
import io
import base64
import logging
from typing import Optional
import sys
import traceback
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    logger.info("Application startup initiated")
    try:
        success = load_model()
        if success:
            logger.info("Startup completed successfully")
        else:
            logger.error("Startup failed - model could not be loaded")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    yield
    logger.info("Application shutdown initiated")

app = FastAPI(
    title="Cloud Detection API",
    description="API for detecting clouds in satellite imagery using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        try:
            base_model = models.resnet34(weights=None)  # Fixed pretrained warning
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base_layers = list(base_model.children())
            self.enc0 = nn.Sequential(*self.base_layers[:3])
            self.enc1 = nn.Sequential(*self.base_layers[3:5])
            self.enc2 = self.base_layers[5]
            self.enc3 = self.base_layers[6]
            self.enc4 = self.base_layers[7]
            self.dec4 = self.contract_block(512, 256)
            self.dec3 = self.contract_block(512, 128)
            self.dec2 = self.contract_block(256, 64)
            self.dec1 = self.contract_block(128, 64)
            self.dec0 = self.contract_block(128, out_channels)

            # Attention gate layers
            self.attn_g_conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
            self.attn_x_conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
            self.attn_f_conv4 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

            self.attn_g_conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
            self.attn_x_conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
            self.attn_f_conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)

            self.attn_g_conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
            self.attn_x_conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
            self.attn_f_conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

            self.attn_g_conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
            self.attn_x_conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
            self.attn_f_conv1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

            self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            
            logger.info("UNET model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing UNET: {e}")
            raise

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec4 = self.dec4(enc4)
        dec3 = self.dec3(torch.cat([dec4, self.attention_gate(enc3, dec4, level=4)], 1))
        dec2 = self.dec2(torch.cat([dec3, self.attention_gate(enc2, dec3, level=3)], 1))
        dec1 = self.dec1(torch.cat([dec2, self.attention_gate(enc1, dec2, level=2)], 1))
        dec0 = self.dec0(torch.cat([dec1, self.attention_gate(enc0, dec1, level=1)], 1))

        dec0 = self.final_conv(dec0)
        dec0 = nn.ReLU()(dec0)
        dec0 = nn.Upsample(size=(384, 384), mode='bilinear', align_corners=True)(dec0)
        return dec0

    def contract_block(self, in_channels, out_channels):
        contract = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Dropout2d(0.1)
        )
        return contract

    def attention_gate(self, x, g, level):
        if level == 4:
            g_conv = self.attn_g_conv4(g)
            x_conv = self.attn_x_conv4(x)
            f_conv = self.attn_f_conv4
        elif level == 3:
            g_conv = self.attn_g_conv3(g)
            x_conv = self.attn_x_conv3(x)
            f_conv = self.attn_f_conv3
        elif level == 2:
            g_conv = self.attn_g_conv2(g)
            x_conv = self.attn_x_conv2(x)
            f_conv = self.attn_f_conv2
        elif level == 1:
            g_conv = self.attn_g_conv1(g)
            x_conv = self.attn_x_conv1(x)
            f_conv = self.attn_f_conv1
        else:
            raise ValueError("Invalid level for attention gate")

        f = nn.ReLU()(g_conv + x_conv)
        f = f_conv(f)
        f = torch.sigmoid(f)
        return x * f

class ProposedSATUNet(UNET):
    def __init__(self, in_channels=4, out_channels=2):
        super(ProposedSATUNet, self).__init__(in_channels, out_channels)
        try:
            self.enhance_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.aux_dec3 = nn.Conv2d(128, out_channels, kernel_size=1)
            self.aux_dec2 = nn.Conv2d(64, out_channels, kernel_size=1)
            self.aux_dec1 = nn.Conv2d(64, out_channels, kernel_size=1)
            self.enc3_proj = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
            self.enc2_proj = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
            self.enc1_proj = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
            
            logger.info("ProposedSATUNet model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ProposedSATUNet: {e}")
            raise

    def forward(self, x):
        x = self.enhance_conv(x)
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(torch.cat([dec4, self.attention_gate(enc3, dec4, level=4)], 1))
        dec3_res = dec3 + F.interpolate(self.enc3_proj(enc3), size=dec3.size()[2:], mode='bilinear', align_corners=True)
        aux3 = F.interpolate(self.aux_dec3(dec3_res), size=x.size()[2:], mode='bilinear', align_corners=True)
        dec2 = self.dec2(torch.cat([dec3_res, self.attention_gate(enc2, dec3_res, level=3)], 1))
        dec2_res = dec2 + F.interpolate(self.enc2_proj(enc2), size=dec2.size()[2:], mode='bilinear', align_corners=True)
        aux2 = F.interpolate(self.aux_dec2(dec2_res), size=x.size()[2:], mode='bilinear', align_corners=True)
        dec1 = self.dec1(torch.cat([dec2_res, self.attention_gate(enc1, dec2_res, level=2)], 1))
        dec1_res = dec1 + F.interpolate(self.enc1_proj(enc1), size=dec1.size()[2:], mode='bilinear', align_corners=True)
        aux1 = F.interpolate(self.aux_dec1(dec1_res), size=x.size()[2:], mode='bilinear', align_corners=True)
        dec0 = self.dec0(torch.cat([dec1_res, self.attention_gate(enc0, dec1_res, level=1)], 1))
        dec0 = self.final_conv(dec0)
        dec0 = nn.Upsample(size=(384, 384), mode='bilinear', align_corners=True)(dec0)
        return dec0, aux3, aux2, aux1

def load_model():
    """Load the trained model"""
    global model
    try:
        logger.info("Starting model initialization...")
        model = ProposedSATUNet(in_channels=4, out_channels=2).to(device)
        
        # Try to load weights if available
        try:
            model_path = os.path.join(os.path.dirname(__file__), r"C:\Users\lenovo\OneDrive\Documents\Cloud Segmentation\server\project_root\Best_model.pth")
            if not os.path.exists(model_path):
                logger.error(f"Model weights file not found at: {model_path}")
                raise FileNotFoundError(f"Model weights file not found at: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            logger.info("Model weights loaded successfully from Best_model.pth")
        except FileNotFoundError as e:
            logger.warning(f"No saved weights found: {e}, using randomly initialized model")
        except Exception as e:
            logger.warning(f"Could not load weights: {e}, using randomly initialized model")
        
        model.eval()
        logger.info(f"Model loaded successfully on device: {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def preprocess_image(image_data: bytes, target_size=(384, 384)) -> np.ndarray:
    """Preprocess image data for model input"""
    try:
        image = Image.open(io.BytesIO(image_data))
        image = image.resize(target_size)
        image_array = np.array(image)
        
        # Normalize to [0, 1]
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        elif image_array.dtype == np.uint16:
            image_array = image_array.astype(np.float32) / 65535.0
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def create_4channel_input(red_data, green_data, blue_data, nir_data):
    """Create 4-channel input from separate band images"""
    try:
        red = preprocess_image(red_data)
        green = preprocess_image(green_data)
        blue = preprocess_image(blue_data)
        nir = preprocess_image(nir_data)
        
        # Stack channels: [H, W, C]
        if len(red.shape) == 3:
            red = red[:, :, 0]  # Take first channel if RGB
        if len(green.shape) == 3:
            green = green[:, :, 0]
        if len(blue.shape) == 3:
            blue = blue[:, :, 0]
        if len(nir.shape) == 3:
            nir = nir[:, :, 0]
        
        # Stack to create 4-channel image
        image_4ch = np.stack([red, green, blue, nir], axis=2)
        
        # Convert to PyTorch format: [C, H, W]
        image_tensor = torch.FloatTensor(image_4ch).permute(2, 0, 1)
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error creating 4-channel input: {e}")
        raise

def create_rgb_input(image_data):
    """Create input from RGB image by duplicating a channel for NIR"""
    try:
        image_array = preprocess_image(image_data)
        
        if len(image_array.shape) == 2:
            # Grayscale image, convert to RGB
            image_array = np.stack([image_array] * 3, axis=2)
        elif image_array.shape[2] == 4:
            # RGBA image, take first 3 channels
            image_array = image_array[:, :, :3]
        
        # Use red channel as NIR approximation
        nir_channel = image_array[:, :, 0]
        
        # Create 4-channel image [R, G, B, NIR]
        image_4ch = np.concatenate([image_array, nir_channel[:, :, np.newaxis]], axis=2)
        
        # Convert to PyTorch format: [C, H, W]
        image_tensor = torch.FloatTensor(image_4ch).permute(2, 0, 1)
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error creating RGB input: {e}")
        raise

def postprocess_prediction(prediction):
    """Convert model prediction to binary mask and metrics"""
    try:
        # Handle tuple output
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        
        # Convert to probability
        if prediction.shape[1] > 1:
            # Multi-class output
            probs = torch.softmax(prediction, dim=1)
            cloud_prob = probs[:, 1]  # Cloud class
        else:
            # Single channel output
            cloud_prob = torch.sigmoid(prediction[:, 0])
        
        # Create binary mask
        binary_mask = (cloud_prob > 0.5).float()
        
        # Calculate metrics
        total_pixels = binary_mask.numel()
        cloud_pixels = binary_mask.sum().item()
        confidence_score = cloud_prob.mean().item()
        
        metrics = {
            "total_pixels": int(total_pixels),
            "cloud_pixels": int(cloud_pixels),
            "cloud_percentage": float(cloud_pixels / total_pixels * 100),
            "non_cloud_pixels": int(total_pixels - cloud_pixels)
        }
        
        return binary_mask, confidence_score, metrics
    except Exception as e:
        logger.error(f"Error in postprocessing: {e}")
        raise

def tensor_to_base64_image(tensor):
    """Convert tensor to base64 encoded image"""
    try:
        # Convert to numpy and scale to 0-255
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        
        mask_np = tensor.cpu().numpy()
        mask_np = (mask_np * 255).astype(np.uint8)
        
        # Create PIL image
        mask_image = Image.fromarray(mask_np, mode='L')
        
        # Convert to base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return mask_base64
    except Exception as e:
        logger.error(f"Error converting tensor to base64: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Cloud Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/predict")
async def predict_get():
    return {"message": "Use POST to /predict with red, green, blue, and nir image files"}

@app.post("/predict")
async def predict(
    red: UploadFile = File(...),
    green: UploadFile = File(...),
    blue: UploadFile = File(...),
    nir: UploadFile = File(...)
):
    """Predict cloud mask from 4-band satellite images and return as PNG image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image data
        red_data = await red.read()
        green_data = await green.read()
        blue_data = await blue.read()
        nir_data = await nir.read()
        
        # Create 4-channel input
        input_tensor = create_4channel_input(red_data, green_data, blue_data, nir_data)
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Postprocess to get binary mask
        binary_mask, _, _ = postprocess_prediction(prediction)
        
        # Convert tensor to numpy and scale to 0-255
        mask_np = binary_mask.squeeze().cpu().numpy() * 255
        mask_np = mask_np.astype(np.uint8)
        
        # Create PIL image
        mask_image = Image.fromarray(mask_np, mode='L')
        
        # Save to bytes buffer
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_rgb")
async def predict_rgb(image: UploadFile = File(...)):
    """Predict cloud mask from RGB image (uses red channel as NIR approximation)"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image data
        image_data = await image.read()
        
        # Create input tensor (RGB + red channel as NIR)
        input_tensor = create_rgb_input(image_data)
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Postprocess prediction
        binary_mask, confidence_score, metrics = postprocess_prediction(prediction)
        
        # Convert mask to base64 image
        mask_base64 = tensor_to_base64_image(binary_mask)
        
        return {
            "prediction_mask": mask_base64,
            "confidence_score": confidence_score,
            "metrics": metrics,
            "input_shape": list(input_tensor.shape),
            "model_device": str(device),
            "note": "NIR channel approximated from red channel"
        }
        
    except Exception as e:
        logger.error(f"RGB prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)