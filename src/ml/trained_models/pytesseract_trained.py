from PIL import Image
from pytesseract import pytesseract


def predict_pytesseract(img):

    path_to_tesseract = r'src/ml/trained_models/Tesseract-OCR/tesseract.exe'
    pytesseract.tesseract_cmd = path_to_tesseract
    img = Image.open(img)
    text = pytesseract.image_to_string(img)
    print(text)
    return text