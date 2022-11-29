# onnx-quantization

## architecture

https://onnxruntime.ai/docs/performance/quantization.html

## official sample

https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu

## requirement

```Per-Channel support with QDQ format requires onnx opset version 13 or above.```

## quantize

```
python3 run.py --input_model mobilenetv2_1.0.opt.onnx --output_model mobilenet_quantized.onnx --calibrate_dataset imagenet_val --per_channel True
```

```
python3 run.py --input_model yolox_tiny.opt.onnx --output_model yolox_tiny_quantized.onnx --calibrate_dataset imagenet_val
```

## output
