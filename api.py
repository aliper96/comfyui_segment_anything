import os
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from PIL import Image
import io
from local_groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import sys
import json

# Add the SAM HQ directory to the path
sam_hq_path = os.path.join(os.path.dirname(__file__), "sam_hq")
if sam_hq_path not in sys.path:
    sys.path.append(sam_hq_path)

from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import build_sam_hq_vit_h

app = FastAPI(title="Grounding SAM API", description="API for object detection and segmentation using Grounding SAM and SAM HQ")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model paths
CHECKPOINTS_DIR = "checkpoints"

# GroundingDINO model
GROUNDING_DINO_CONFIG_PATH = os.path.join(CHECKPOINTS_DIR, "GroundingDINO_SwinT_OGC.cfg.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "groundingdino_swint_ogc.pth")

# SAM model
SAM_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "sam_vit_h_4b8939.pth")
SAM_HQ_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "sam_hq_vit_h.pth")

# Initialize models
grounding_dino_model = None
sam_predictor = None
sam_hq_predictor = None

def initialize_models():
    global grounding_dino_model, sam_predictor, sam_hq_predictor
    
    # Initialize GroundingDINO
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    
    # Initialize SAM
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    
    # Initialize SAM HQ using the correct builder function
    sam_hq = build_sam_hq_vit_h(checkpoint=SAM_HQ_CHECKPOINT_PATH)
    sam_hq.to(device=DEVICE)
    sam_hq_predictor = SamPredictorHQ(sam_hq)

@app.on_event("startup")
async def startup_event():
    initialize_models()

# def process_image(image_bytes: bytes) -> np.ndarray:
#     # Convert bytes to numpy array
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def process_image(image_bytes: bytes) -> np.ndarray:
    try:
        # Convertir bytes a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Error al decodificar la imagen. Verifica el formato del archivo.")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"❌ Error en el procesamiento de imagen: {e}")
        raise


def numpy_to_python(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj


@app.post("/detect_objects")
async def detect_objects(
        image: UploadFile = File(...),
        prompt: Optional[str] = Form(None),
        use_hq: bool = Form(False),
        confidence_threshold: float = Form(0.35),
        box_threshold: float = Form(0.3)
):
    try:
        print(f"Received request with image: {image.filename}")
        image_bytes = await image.read()
        print(f"Image size: {len(image_bytes)} bytes")
        image_array = process_image(image_bytes)
        print(f"Image processed, shape: {image_array.shape}")

        # If no prompt is provided, use automatic detection
        if not prompt:
            prompt = "all objects"
        print(f"Using prompt: {prompt}")

        # Get detections from GroundingDINO
        detections = grounding_dino_model.predict_with_classes(
            image=image_array,
            classes=[prompt],
            box_threshold=box_threshold,
            text_threshold=confidence_threshold
        )

        boxes = detections.xyxy
        print(f"Detected {len(boxes)} boxes")

        if len(boxes) == 0:
            return JSONResponse(content={
                "success": True,
                "objects": []
            })

        labels = detections.class_id
        scores = detections.confidence

        # Use appropriate SAM predictor
        predictor = sam_hq_predictor if use_hq else sam_predictor
        predictor.set_image(image_array)

        results = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            print(f"Processing box {i}: {box.tolist()}")

            # Get mask from SAM
            try:
                masks, _, _ = predictor.predict(
                    box=box,
                    multimask_output=False
                )

                if masks is None or len(masks) == 0:
                    print(f"No masks returned for box {i}")
                    continue

                # Convert mask to binary format and get contours
                mask = masks[0].astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours is None or len(contours) == 0:
                    print(f"No contours found for box {i}")
                    contours_list = []
                else:
                    # Convert contours to list format
                    contours_list = [cont.reshape(-1).tolist() for cont in contours]

                result_item = {
                    "label": label,
                    "confidence": score,
                    "bbox": box.tolist(),
                    "contours": contours_list
                }

                # Convert all numpy types to Python native types
                result_item = numpy_to_python(result_item)
                results.append(result_item)

            except Exception as e:
                print(f"Error processing box {i}: {e}")

        print(f"Returning {len(results)} object results")
        return JSONResponse(content={
            "success": True,
            "objects": results
        })

    except Exception as e:
        import traceback
        print(f"Error in detect_objects: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 