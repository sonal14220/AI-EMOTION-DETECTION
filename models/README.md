# Emotion Detection Models

This directory contains the emotion detection models used by the application.

## Model Requirements

The application expects an ONNX model file named `emotion_model.onnx` with the following specifications:

- **Input**: 1x3x224x224 (batch_size, channels, height, width)
- **Output**: 1x7 (7 emotion classes)
- **Emotion Classes**: ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

## Getting a Model

### Option 1: Use a Pre-trained Model
You can download a pre-trained emotion detection model from:
- [Hugging Face Model Hub](https://huggingface.co/models?search=emotion+detection)
- [ONNX Model Zoo](https://github.com/onnx/models)

### Option 2: Convert Your Own Model

#### From TensorFlow/Keras:
```python
import tensorflow as tf
import tf2onnx

# Load your trained model
model = tf.keras.models.load_model('your_model.h5')

# Convert to ONNX
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)
output_path = "models/emotion_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
```

#### From PyTorch:
```python
import torch
import torch.onnx

# Load your trained model
model = YourEmotionModel()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/emotion_model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},
                  'output': {0: 'batch_size'}}
)
```

## Model Testing

You can test your model using the provided test script:

```bash
python scripts/test_model.py
```

## Performance Optimization

For better performance, consider:
1. **Quantization**: Convert to INT8 for faster inference
2. **Model Pruning**: Remove unnecessary weights
3. **TensorRT**: Use NVIDIA's TensorRT for GPU acceleration

## Example Model Download

```bash
# Download a sample model (replace with actual URL)
wget https://example.com/emotion_model.onnx -O models/emotion_model.onnx
```

## Troubleshooting

If you encounter issues:
1. Verify the model input/output shapes
2. Check that the model file is in the correct location
3. Ensure the model supports the expected emotion classes
4. Test the model independently before using with the web app 