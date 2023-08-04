import io
from pathlib import Path

import numpy as np
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from PIL import Image

def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def tesseract_ocr_image(image_data):
    image_data = image_to_byte_array(image_data)
    image_stream = io.BytesIO(image_data)
    with Image.open(image_stream) as img:
        text = pytesseract.image_to_string(img)
    return text

def easyocr_ocr_image(image_path):
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_path)
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def paddle_ocr_image(image):
    det_model_dir = str(Path(__file__).parent / '/paddle_files_dir/det')
    rec_model_dir = str(Path(__file__).parent.parent.parent.parent / 'paddle_files_dir/rec')
    cls_model_dir = str(Path(__file__).parent.parent.parent.parent / 'paddle_files_dir/cls')
    image = image_to_byte_array(image)
    ocr_reader = PaddleOCR(det_model_dir=det_model_dir, rec_model_dir=rec_model_dir, cls_model_dir=cls_model_dir)
    image_stream = io.BytesIO(image)
    with Image.open(image_stream) as img:
        np_img = np.array(img)
        result = ocr_reader.ocr(np_img)
        text = '\n'.join([word[1][0] for word in result[0]])
    return text