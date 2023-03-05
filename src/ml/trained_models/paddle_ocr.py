import io
from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


def ocr_image(image):
    det_model_dir = str(Path(__file__).parent / 'paddle_ocr/det')
    rec_model_dir = str(Path(__file__).parent / 'paddle_ocr/rec')
    cls_model_dir = str(Path(__file__).parent / 'paddle_ocr/cls')
    ocr_reader = PaddleOCR(det_model_dir=det_model_dir, rec_model_dir=rec_model_dir, cls_model_dir=cls_model_dir)
    image_data = image.file.read()
    image_stream = io.BytesIO(image_data)
    with Image.open(image_stream) as img:
        np_img = np.array(img)
        result = ocr_reader.ocr(np_img)
        text = '\n'.join([word[1][0] for word in result[0]])
    return text
