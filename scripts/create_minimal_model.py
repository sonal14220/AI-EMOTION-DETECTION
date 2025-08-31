#!/usr/bin/env python3
"""
Create Minimal Working ONNX Model
=================================
Creates a minimal ONNX model that will definitely work.
"""

import numpy as np
from pathlib import Path

def create_minimal_model():
    """Create a minimal working ONNX model"""
    
    try:
        import onnx
        from onnx import helper, numpy_helper
        
        print("🔧 Creating minimal working ONNX model...")
        
        # Minimal model structure
        input_shape = [1, 3, 224, 224]
        output_shape = [1, 7]
        
        # Input
        input_tensor = helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, input_shape
        )
        
        # Output
        output_tensor = helper.make_tensor_value_info(
            'output', onnx.TensorProto.FLOAT, output_shape
        )
        
        # Create a minimal model with just one operation
        # We'll use a simple identity-like operation that outputs 7 values
        
        # Create random weights for a simple transformation
        weight = np.random.randn(7, 3*224*224).astype(np.float32) * 0.01
        bias = np.random.randn(7).astype(np.float32) * 0.01
        
        weight_tensor = numpy_helper.from_array(weight, name='weight')
        bias_tensor = numpy_helper.from_array(bias, name='bias')
        
        # Flatten input
        nodes = [
            helper.make_node(
                'Flatten', ['input'], ['flattened'],
                name='flatten'
            )
        ]
        
        # Simple linear transformation
        nodes.append(helper.make_node(
            'Gemm', ['flattened', 'weight', 'bias'], ['output'],
            name='linear', alpha=1.0, beta=1.0
        ))
        
        # Create graph
        graph = helper.make_graph(
            nodes, 'minimal_emotion_detection',
            [input_tensor], [output_tensor],
            [weight_tensor, bias_tensor]
        )
        
        # Create model with IR version 10
        model = helper.make_model(
            graph,
            producer_name='MinimalEmotionDetection',
            producer_version='1.0',
            opset_imports=[helper.make_opsetid('', 10)]
        )
        
        # Force IR version to 10
        model.ir_version = 10
        
        # Save model
        model_path = Path("models/emotion_model.onnx")
        onnx.save(model, str(model_path))
        
        print(f"✅ Created minimal working model at: {model_path}")
        print("📊 This model will load correctly and simulate emotion detection")
        print("🎯 For real 90%+ accuracy, replace with a trained model")
        
        return True
        
    except ImportError:
        print("❌ ONNX not available")
        return False
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False

def test_model():
    """Test the created model"""
    print("\n🧪 Testing the model...")
    
    try:
        import onnxruntime as ort
        
        # Load the model
        session = ort.InferenceSession("models/emotion_model.onnx")
        print("✅ Model loaded successfully!")
        
        # Test inference
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {'input': input_data})
        
        print("✅ Inference successful!")
        print(f"📊 Output shape: {outputs[0].shape}")
        print(f"📊 Output values: {outputs[0][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main function"""
    print("🎯 Creating Minimal Working ONNX Model")
    print("=" * 40)
    
    success = create_minimal_model()
    
    if success:
        print("\n🎉 Model created successfully!")
        
        # Test the model
        if test_model():
            print("\n✅ Model is working correctly!")
        else:
            print("\n⚠️  Model test failed, but app will run in demo mode")
        
        print("\n📋 Next steps:")
        print("1. Run the app: uvicorn app.main:app --reload")
        print("2. Open: http://localhost:8000/")
        print("3. Accept consent and start detection!")
        
    else:
        print("\n❌ Failed to create model")

if __name__ == "__main__":
    main() 