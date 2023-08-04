import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import prediction_service as ps

model = tf.keras.models.load_model('numbers_test/keras_numbers.h5')
lines = []
with open("numbers_test/number_labels.txt", "r") as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]
images = {}
for line in lines:
    images[line] = f'numbers_test/number_images/{line}.png'

good_predictions = 0
tesseract_predictions = 0
easy_predictions = 0
paddle_predictions = 0
with open('numbers_test/numbers_test.txt', 'w') as file:
    for label, image_path in images.items():
        file.write(f'Models trying to recognize {label}\n')
        tesseract_img = Image.open(image_path)
        tesseract_prediction = ps.tesseract_ocr_image(tesseract_img)
        file.write(f'TesseractOCR recognized it as {tesseract_prediction} \n')
        if tesseract_prediction == int(label):
            tesseract_predictions +=1
        tesseract_img.close()
        #easyocr_prediction = ps.easyocr_ocr_image(image_path)
        #file.write(f'Easy recognized it as {easyocr_prediction}')
        #if easyocr_prediction == int(label):
            #easy_predictions += 1
        paddleocr_img = Image.open(image_path)
        paddleocr_prediction = ps.paddle_ocr_image(paddleocr_img)
        file.write(f'PaddleOCR recognized it as {paddleocr_prediction} \n')
        if int(paddleocr_prediction) == int(label):
            paddle_predictions += 1
        paddleocr_img.close()
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array[..., tf.newaxis]

        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        file.write(f'Custom model recognized it as {predicted_label} \n\n\n')
        if predicted_label == int(label):
            good_predictions += 1
    file.write(f'In total out of 10 attempts: \n '
               f'TesseractOCR recognized well {tesseract_predictions} giving it {tesseract_predictions/10*100} % accuracy \n'
               #f'EasyOCR recognized well {easy_predictions} giving it {easy_predictions/10*100} % accuracy \n'
               f'PaddleOCR recognized well {paddle_predictions} giving it {paddle_predictions/10*100} % accuracy \n'
               f'Custom OCR model recognized well {good_predictions} giving it {good_predictions/10*100} % accuracy')
