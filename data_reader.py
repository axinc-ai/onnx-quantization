import numpy
import onnxruntime
import os
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0, model_path: str=None):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        if "yolox" in model_path:
            pillow_img = Image.new("RGB", (width, height))
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            input_data = numpy.float32(pillow_img) # 0 - 255
            input_data = input_data[:,:,::-1] # RGB -> BGR
        elif "yolov3" in model_path:
            pillow_img = Image.new("RGB", (width, height)) # RGB
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            input_data = numpy.float32(pillow_img) / 255.0 # 0 - 1
        else:
            pillow_img = Image.new("RGB", (width, height)) # RGB
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            input_data = numpy.float32(pillow_img) - numpy.array(
                [123.68, 116.78, 103.94], dtype=numpy.float32
            ) # -128 - 127
        nhwc_data = numpy.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape
        if "yolov3" in model_path:
            width = 416
            height = 416

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0, model_path = model_path
        )
        self.input_name = session.get_inputs()[0].name
        self.shape_nam = None
        if "yolov3" in model_path:
            self.shape_name = session.get_inputs()[1].name
            self.width = width
            self.height = height
            self.iou_name = session.get_inputs()[2].name
            self.threshold_name = session.get_inputs()[3].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            if self.shape_name:
                shape_data = numpy.array([self.height, self.width], dtype='float32').reshape(1, 2)
                iou_data = numpy.array([numpy.random.rand()], dtype='float32').reshape(1)
                threshold_data = numpy.array([numpy.random.rand()], dtype='float32').reshape(1)
                self.enum_data = iter(
                    [{self.input_name: nhwc_data, self.shape_name: shape_data, self.iou_name: iou_data, self.threshold_name: threshold_data} for nhwc_data in self.nhwc_data_list]
                )
            else:
                self.enum_data = iter(
                    [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
                )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
