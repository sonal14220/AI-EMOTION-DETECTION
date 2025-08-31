#!/usr/bin/env python3
"""
Download a sample emotion detection model for testing
"""

import os
import sys
import requests
from pathlib import Path

def download_model():
    """Download a sample emotion detection model"""
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "emotion_model.onnx"
    
    if model_path.exists():
        print(f"✅ Model already exists at {model_path}")
        return True
    
    # Sample model URLs (you can replace these with actual model URLs)
    model_urls = [
        "https://github.com/onnx/models/raw/main/vision/emotion_ferplus/model/emotion-ferplus-8.onnx",
        # Add more URLs as needed
    ]
    
    print("📥 Downloading sample emotion detection model...")
    print("Note: This is a placeholder. You'll need to provide your own model.")
    print("\nTo get a real emotion detection model:")
    print("1. Visit: https://huggingface.co/models?search=emotion+detection")
    print("2. Download an ONNX model with 7 emotion classes")
    print("3. Place it in the models/ directory as 'emotion_model.onnx'")
    print("4. Ensure it has input shape (1,3,224,224) and output shape (1,7)")
    
    # Create a dummy model file for testing
    print("\nCreating dummy model file for testing...")
    
    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        
        # Create a simple dummy model with compatible opset
        input_shape = [1, 3, 224, 224]
        output_shape = [1, 7]
        
        # Create input
        input_tensor = helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, input_shape
        )
        
        # Create output
        output_tensor = helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, output_shape
        )
        
        # Create a simple identity-like operation
        # This is just for testing - not a real emotion detection model
        identity_node = helper.make_node(
            'Identity',
            inputs=['input'],
            outputs=['output'],
            name='identity'
        )
        
        # Create graph
        graph = helper.make_graph(
            [identity_node],
            'emotion_detection',
            [input_tensor],
            [output_tensor]
        )
        
        # Create model with compatible opset version
        model = helper.make_model(graph, producer_name='dummy_emotion_model', opset_imports=[
            helper.make_opsetid('', 10)  # Use opset 10 for compatibility
        ])
        
        # Save model
        with open(model_path, 'wb') as f:
            f.write(model.SerializeToString())
        
        print(f"✅ Created dummy model at {model_path}")
        print("⚠️  This is NOT a real emotion detection model!")
        print("   It will return random predictions for testing purposes only.")
        print("   Please replace with a real trained model for actual use.")
        
        return True
        
    except ImportError:
        print("❌ ONNX not available. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
            print("✅ ONNX installed. Please run this script again.")
        except Exception as e:
            print(f"❌ Failed to install ONNX: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to create dummy model: {e}")
        return False

def main():
    """Main function"""
    print("🤖 Emotion Detection Model Downloader")
    print("=" * 50)
    
    success = download_model()
    
    if success:
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Test the model: python scripts/test_model.py")
        print("2. Run the app: uvicorn app.main:app --reload")
        print("3. Open: http://localhost:8000/static/")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 