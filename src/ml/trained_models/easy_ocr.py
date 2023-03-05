import easyocr


def ocr_image(image):
    reader = easyocr.Reader(['en'])
    image_data = image.file.read()
    result = reader.readtext(image_data)
    text = ' '.join([item[1] for item in result])
    return text
