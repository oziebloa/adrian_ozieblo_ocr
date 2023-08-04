import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import prediction_service as ps

def label_to_char(predicted_label):
    if predicted_label < 10:
        return str(predicted_label)  # For digits 0-9
    else:
        return chr(predicted_label - 10 + ord('A'))  # For letters A-Z


model_cnn_kaggle = tf.keras.models.load_model('characters_test/cnn_kaggle.h5')
model_crnn_kaggle = tf.keras.models.load_model('characters_test/crnn_kaggle.h5')
model_emnist_cnn = tf.keras.models.load_model('characters_test/emnist_cnn.h5')
model_emnist_crnn = tf.keras.models.load_model('characters_test/emnist_crnn.h5')
lines = []
with open("characters_test/character_labels.txt", "r") as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]
images = {}
for line in lines:
    images[line] = f'characters_test/character_images/{line}.png'

good_predictions_cnn_kaggle = 0
good_predictions_crnn_kaggle = 0
good_predictions_emnist_cnn = 0
good_predictions_emnist_crnn = 0
tesseract_predictions = 0
easy_predictions = 0
paddle_predictions = 0
with open('characters_test/character_test.txt', 'w') as file:
    for label, image_path in images.items():
        file.write(f'Models trying to recognize {label}\n')
        tesseract_img = Image.open(image_path)
        tesseract_prediction = ps.tesseract_ocr_image(tesseract_img)
        file.write(f'TesseractOCR recognized it as {tesseract_prediction} \n')
        if tesseract_prediction == label:
            tesseract_predictions +=1
        tesseract_img.close()
        #easyocr_prediction = ps.easyocr_ocr_image(image_path)
        #file.write(f'Easy recognized it as {easyocr_prediction}')
        #if easyocr_prediction == int(label):
            #easy_predictions += 1
        paddleocr_img = Image.open(image_path)
        paddleocr_prediction = ps.paddle_ocr_image(paddleocr_img)
        file.write(f'PaddleOCR recognized it as {paddleocr_prediction} \n')
        if str(paddleocr_prediction) == str(label):
            paddle_predictions += 1
        paddleocr_img.close()
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array[..., tf.newaxis]

        prediction_cnn_kaggle = model_cnn_kaggle.predict(img_array)
        predicted_label = np.argmax(prediction_cnn_kaggle)
        predicted_label = label_to_char(predicted_label)
        file.write(f'CNN_Kaggle model recognized it as {predicted_label} \n')
        if str(predicted_label) == str(label):
            good_predictions_cnn_kaggle += 1
        prediction_crnn_kaggle = model_crnn_kaggle.predict(img_array)
        predicted_label = np.argmax(prediction_crnn_kaggle)
        predicted_label = label_to_char(predicted_label)
        file.write(f'CRNN_Kaggle model recognized it as {predicted_label} \n')
        if str(predicted_label) == str(label):
            good_predictions_crnn_kaggle += 1
        prediction_emnist_cnn = model_emnist_cnn.predict(img_array)
        predicted_label = np.argmax(prediction_emnist_cnn)
        predicted_label = label_to_char(predicted_label)
        file.write(f'CNN_Emnist model recognized it as {predicted_label} \n')
        if str(predicted_label) == str(label):
            good_predictions_emnist_cnn += 1
        prediction_emnist_crnn = model_emnist_crnn.predict(img_array)
        predicted_label = np.argmax(prediction_emnist_crnn)
        predicted_label = label_to_char(predicted_label)
        file.write(f'CRNN_Emnist model recognized it as {predicted_label} \n\n\n')
        if str(predicted_label) == str(label):
            good_predictions_emnist_crnn += 1
    file.write(f'In total out of 36 attempts: \n '
               f'TesseractOCR recognized well {tesseract_predictions} giving it {tesseract_predictions/36*100} % accuracy \n'
               #f'EasyOCR recognized well {easy_predictions} giving it {easy_predictions/10*100} % accuracy \n'
               f'PaddleOCR recognized well {paddle_predictions} giving it {paddle_predictions/36*100} % accuracy \n'
               f'CNN_Kaggle OCR model recognized well {good_predictions_cnn_kaggle} giving it {good_predictions_cnn_kaggle/36*100} % accuracy \n'
               f'CRNN_Kaggle OCR model recognized well {good_predictions_crnn_kaggle} giving it {good_predictions_crnn_kaggle/36*100} % accuracy \n'
               f'CNN_Emnist OCR model recognized well {good_predictions_emnist_cnn} giving it {good_predictions_emnist_cnn/36*100} % accuracy \n'
               f'CRNN_Emnist OCR model recognized well {good_predictions_emnist_crnn} giving it {good_predictions_emnist_crnn/36*100} % accuracy')
