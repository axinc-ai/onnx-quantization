# onnx-quantization

This is a example to quantize onnx. The input is onnx of float. Quantization is done using onnxruntime. The output is onnx of int8.

## Architecture

```mermaid
classDiagram
`ONNX (int8)` <|-- `ONNX Runtime` : Quantized model
`ONNX Runtime` <|-- `ONNX (float)` : Input model
`ONNX Runtime` <|-- `Images` : Calibration images
`ONNX Runtime` : quantize_static API
`ONNX (float)` : FLoat model
`ONNX (int8)` : Int8 model
`Images` : Images
```

## Calibration images

Place imagenet validation images in the imagenet_val folder to improve quantization accuracy. The default is to quantize using only 2 images, which is less accurate.

## Command

Quantization can be performed with the following command. 

MobileNetV2

```
python3 quantize.py --input_model mobilenetv2_1.0.opt.onnx --output_model mobilenet_quantized.onnx --calibrate_dataset imagenet_val --per_channel True
```

YOLOX Tiny

```
python3 quantize.py --input_model yolox_tiny.opt.onnx --output_model yolox_tiny_quantized.onnx --calibrate_dataset imagenet_val
```

## Output

- [yolox_tiny_quantized.onnx](yolox_tiny_quantized.onnx)
- [mobilenev2_quantized.onnx](mobilenev2_quantized.onnx)

## Limitation

Per-Channel support with QDQ format requires onnx opset version 13 or above.

## Reference

Official sample
https://onnxruntime.ai/docs/performance/quantization.html

Official document
https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu
