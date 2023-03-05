import io

from PIL import Image

import pytesseract


def ocr_image(image):
    image_data = image.file.read()
    image_stream = io.BytesIO(image_data)
    # pytesseract.pytesseract.tesseract_cmd = Path(__file__).parent / 'Tesseract-OCR/tesseract.exe'
    with Image.open(image_stream) as img:
        text = pytesseract.image_to_string(img)
        print('done')
    return text
