from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import base64
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import json
import logging
from typing import List, Dict, Any
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Emotion Detection API",
    description="Real-time emotion detection using computer vision and deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5
)

# Initialize ONNX runtime session
try:
    session = ort.InferenceSession("models/emotion_model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info("ONNX model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    logger.info("Running in demo mode with dummy predictions")
    session = None
    input_name = None
    output_name = None

# Emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class EmotionDetector:
    def __init__(self):
        self.face_detection = mp_face_detection
        self.session = session
        self.input_name = input_name
        self.emotion_labels = EMOTION_LABELS
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input"""
        # Resize to 224x224 (standard input size)
        face_resized = cv2.resize(face_img, (224, 224))
        # Convert to float32 and normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        # Convert to NCHW format (batch, channels, height, width)
        face_tensor = np.transpose(face_normalized, (2, 0, 1))
        # Add batch dimension
        face_batch = np.expand_dims(face_tensor, axis=0)
        return face_batch
    
    def detect_emotion(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Detect emotion in face image"""
        if self.session is None:
            # Return dummy predictions for demo mode
            import random
            dummy_predictions = [random.random() for _ in range(7)]
            # Normalize to sum to 1
            total = sum(dummy_predictions)
            dummy_predictions = [p/total for p in dummy_predictions]
            emotion_idx = np.argmax(dummy_predictions)
            confidence = float(dummy_predictions[emotion_idx])
            emotion = self.emotion_labels[emotion_idx]
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "predictions": dummy_predictions,
                "inference_time_ms": 50.0,  # Simulated inference time
                "demo_mode": True
            }
        
        try:
            # Preprocess face
            input_tensor = self.preprocess_face(face_img)
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            inference_time = time.time() - start_time
            
            # Get predictions
            predictions = outputs[0][0]
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx])
            emotion = self.emotion_labels[emotion_idx]
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "predictions": predictions.tolist(),
                "inference_time_ms": round(inference_time * 1000, 2)
            }
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return {"error": str(e)}
    
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process frame and return emotion detections for all faces"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        detections = []
        if results.detections:
            height, width = frame.shape[:2]
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w > 0 and h > 0:
                    # Extract face region
                    face_img = rgb_frame[y:y+h, x:x+w]
                    
                    # Detect emotion
                    emotion_result = self.detect_emotion(face_img)
                    
                    detection_result = {
                        "bbox": [x, y, x + w, y + h],
                        "emotion": emotion_result
                    }
                    detections.append(detection_result)
        
        return detections

# Initialize emotion detector
emotion_detector = EmotionDetector()

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse("web-client/index.html")

@app.get("/app.js")
async def get_app_js():
    """Serve the JavaScript file"""
    return FileResponse("web-client/app.js")

@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "AI Emotion Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "timestamp": time.time()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time emotion detection"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive base64 encoded image
            data = await websocket.receive_text()
            
            try:
                # Parse base64 image
                if "," in data:
                    header, b64_data = data.split(",", 1)
                else:
                    b64_data = data
                
                # Decode base64 to image
                img_bytes = base64.b64decode(b64_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Invalid image data"})
                    continue
                
                # Process frame for emotion detection
                detections = emotion_detector.process_frame(frame)
                
                # Send results back to client
                response = {
                    "detections": detections,
                    "timestamp": time.time()
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 