#!/usr/bin/env python3
"""
Test script for emotion detection model
"""

import sys
import os
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("Testing model loading...")
    
    model_path = "models/emotion_model.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please add your emotion_model.onnx file to the models/ directory")
        return False
    
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"✅ Model loaded successfully")
        print(f"   Input name: {input_name}")
        print(f"   Output name: {output_name}")
        
        # Check input shape
        input_shape = session.get_inputs()[0].shape
        print(f"   Input shape: {input_shape}")
        
        # Check output shape
        output_shape = session.get_outputs()[0].shape
        print(f"   Output shape: {output_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_model_inference():
    """Test model inference with dummy data"""
    print("\nTesting model inference...")
    
    model_path = "models/emotion_model.onnx"
    if not os.path.exists(model_path):
        print("❌ Model file not found")
        return False
    
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create dummy input (1x3x224x224)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})
        predictions = outputs[0][0]
        
        print(f"✅ Inference successful")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Output sum: {np.sum(predictions):.4f}")
        print(f"   Output range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        
        # Check if output has 7 classes
        if len(predictions) == 7:
            print(f"✅ Output has correct number of emotion classes (7)")
        else:
            print(f"❌ Expected 7 emotion classes, got {len(predictions)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def test_face_preprocessing():
    """Test face preprocessing pipeline"""
    print("\nTesting face preprocessing...")
    
    try:
        # Create a dummy face image (RGB, 100x100)
        face_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Preprocess (resize to 224x224, normalize to [0,1], convert to NCHW)
        face_resized = cv2.resize(face_img, (224, 224))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_tensor = np.transpose(face_normalized, (2, 0, 1))
        face_batch = np.expand_dims(face_tensor, axis=0)
        
        print(f"✅ Preprocessing successful")
        print(f"   Input shape: {face_img.shape}")
        print(f"   Output shape: {face_batch.shape}")
        print(f"   Value range: [{np.min(face_batch):.4f}, {np.max(face_batch):.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False

def test_end_to_end():
    """Test complete pipeline with dummy data"""
    print("\nTesting end-to-end pipeline...")
    
    model_path = "models/emotion_model.onnx"
    if not os.path.exists(model_path):
        print("❌ Model file not found")
        return False
    
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create dummy face image
        face_img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        
        # Preprocess
        face_resized = cv2.resize(face_img, (224, 224))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_tensor = np.transpose(face_normalized, (2, 0, 1))
        face_batch = np.expand_dims(face_tensor, axis=0)
        
        # Run inference
        outputs = session.run([output_name], {input_name: face_batch})
        predictions = outputs[0][0]
        
        # Get emotion label
        emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        emotion_idx = np.argmax(predictions)
        emotion = emotion_labels[emotion_idx]
        confidence = float(predictions[emotion_idx])
        
        print(f"✅ End-to-end test successful")
        print(f"   Detected emotion: {emotion}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   All predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Emotion Detection Model Test Suite")
    print("=" * 50)
    
    tests = [
        test_model_loading,
        test_model_inference,
        test_face_preprocessing,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your model is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 