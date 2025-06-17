from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import logging
import requests
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
device = torch.device("cpu")
model = None

class ProposedSATUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.final = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.final(x)
        return x

def load_model():
    global model
    try:
        logger.info("Starting model initialization...")
        model = ProposedSATUNet(in_channels=4, out_channels=2).to(device)
        model_path = "/tmp/Best_model.pth"
        if not os.path.exists(model_path):
            logger.info("Downloading model weights...")
            url = "YOUR_CLOUD_STORAGE_URL/Best_model.pth"  # Replace with Google Drive URL
            with open(model_path, "wb") as f:
                f.write(requests.get(url).content)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.on_event("startup")
async def startup():
    load_model()

@app.post("/predict")
async def predict(
    red: UploadFile = File(...),
    green: UploadFile = File(...),
    blue: UploadFile = File(...),
    nir: UploadFile = File(...)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        images = [
            Image.open(io.BytesIO(await f.read())).resize((384, 384)).convert("L")
            for f in [red, green, blue, nir]
        ]
        input_array = np.stack([np.array(img, dtype=np.float32) / 255.0 for img in images], axis=0)
        input_tensor = torch.tensor(input_array).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        output = torch.softmax(output, dim=1)[:, 1].squeeze(0).cpu().numpy()
        output_img = Image.fromarray((output * 255).astype(np.uint8))
        buffer = io.BytesIO()
        output_img.save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vercel serverless handler
def handler(event, context):
    from mangum import Mangum
    asgi_handler = Mangum(app)
    return asgi_handler(event, context)