import glob
import os

from difflib import get_close_matches, SequenceMatcher
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import prediction_service as ps

def sort_contours(contours):
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                            key=lambda b: b[1][1]))
    sorted_contours = []
    y_threshold = 20
    for i in range(len(boundingBoxes)):
        if i == 0:
            line = [(boundingBoxes[i], contours[i])]
        else:
            if abs(boundingBoxes[i][1] - boundingBoxes[i - 1][1]) <= y_threshold:
                line.append((boundingBoxes[i], contours[i]))
            else:
                line.sort(key=lambda b: b[0][0])
                sorted_contours.extend([contour for _, contour in line])
                line = [(boundingBoxes[i], contours[i])]
    line.sort(key=lambda b: b[0][0])
    sorted_contours.extend([contour for _, contour in line])
    return sorted_contours



def segment_characters(image_path, label):
    img = cv2.imread(image_path)
    scaleFactor = 5
    inputImage = cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    windowSize = 41
    constantValue = 8
    binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                        windowSize, constantValue)
    kernel = np.ones((2, 2), np.uint8)
    binaryImage = cv2.erode(binaryImage, kernel, iterations=1)
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sort_contours(contours)
    fake_index = 0
    for _, c in enumerate(contours):
        boundRect = cv2.boundingRect(c)
        rectX = int(boundRect[0])
        rectY = int(boundRect[1])
        rectWidth = int(boundRect[2])
        rectHeight = int(boundRect[3])

        if label[fake_index] in [' ', '\n']:
            fake_index += 1
            continue

        referenceRatio = 1.0
        contourRatio = rectWidth / rectHeight
        epsilon = 1.6
        ratioDifference = abs(referenceRatio - contourRatio)
        if rectWidth / rectHeight < 0.1 or ratioDifference > epsilon:
            continue
        minArea = 40 * scaleFactor
        maxArea = 80 * minArea
        if minArea <= rectWidth * rectHeight < maxArea:
            croppedChar = inputImage[rectY:rectY + rectHeight, rectX:rectX + rectWidth]
            if croppedChar.shape[0] <= 18 and croppedChar.shape[1] <= 18:
                continue
            margin = 1.17
            if rectHeight > rectWidth:
                size = int(margin * rectHeight)
            else:
                size = int(margin * rectWidth)
            empty_image = 255 * np.ones(shape=[size, size, 3], dtype=np.uint8)
            y_pos = (size - rectHeight) // 2
            x_pos = (size - rectWidth) // 2
            empty_image[y_pos:y_pos + rectHeight, x_pos:x_pos + rectWidth] = croppedChar
            filename = os.path.join('./tmp', f'{str(fake_index)}_{label[fake_index]}.png')
            fake_index += 1
            cv2.imwrite(filename, empty_image)


def clear_directory(dir_path='./tmp'):
    if not os.path.exists(dir_path):
        print(f"The directory {dir_path} does not exist.")
        return
    files = glob.glob(os.path.join(dir_path, '*'))
    for f in files:
        os.remove(f)


def label_to_char(label):
    character_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    label_to_char_map = dict(enumerate(character_map))
    return label_to_char_map[label]


def preprocess_characters(tmp_dir='./tmp'):
    preprocessed_characters = []
    filenames = os.listdir(tmp_dir)
    filenames = sorted(filenames, key=lambda filename: int(filename.split('_')[0]))
    for filename in filenames:
        img = Image.open(f'tmp/{filename}').convert('L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img) / 255.0
        img_array = img_array[..., tf.newaxis]
        preprocessed_characters.append(img_array)
    return preprocessed_characters, filenames


def predict_characters(preprocessed_characters, model):
    recognized_text = ''
    for croppedChar in preprocessed_characters:
        prediction = model.predict(np.array([croppedChar]))
        predicted_label = np.argmax(prediction)
        predicted_char = label_to_char(predicted_label)
        recognized_text += predicted_char
    return recognized_text



lines = []
with open("words_real_test/labels.txt", "r") as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]
total_pred_len = 0
with open('words_real_test/full_sentences_test.txt', 'w') as file:
    good_characters = 0
    good_characters_case_insensitive = 0
    tesseract_predictions = 0
    easy_predictions = 0
    paddle_predictions = 0
    all_characters = 0
    for element in lines:
        local_good = 0
        stripped_element = element.replace(' ', '').replace('\n', '')
        img = f'words_real_test/images/{element}.png'
        all_characters += len(element)
        tesseract_img = Image.open(img)
        tesseract_prediction = ps.tesseract_ocr_image(tesseract_img)
        min_length = min(len(tesseract_prediction), len(element))
        for i in range(min_length):
            if str(tesseract_prediction[i]) == element[i]:
                tesseract_predictions += 1
        tesseract_img.close()

        easyocr_prediction = ps.easyocr_ocr_image(img)
        easyocr_prediction = str(easyocr_prediction[0][1])
        easyocr_prediction = easyocr_prediction.replace(' ', '').replace('\n', '')
        min_length = min(len(easyocr_prediction), len(stripped_element))
        for i in range(min_length):
            if str(easyocr_prediction[i]) == stripped_element[i]:
                easy_predictions += 1

        paddleocr_img = Image.open(img)
        paddleocr_prediction = ps.paddle_ocr_image(paddleocr_img)
        paddleocr_prediction = paddleocr_prediction.replace(' ', '').replace('\n', '')
        min_length = min(len(paddleocr_prediction), len(stripped_element))
        for i in range(min_length):
            if str(paddleocr_prediction[i]) == stripped_element[i]:
                paddle_predictions += len(paddleocr_prediction[i])
        paddleocr_img.close()

        segment_characters(img, stripped_element)
        processed_images, filenames = preprocess_characters()
        loaded_model = tf.keras.models.load_model('words_test/emnist_deep.h5')
        text = predict_characters(processed_images, loaded_model)
        min_length = min(len(text), len(stripped_element))
        for i in range(min_length):
            if str(text[i]) == stripped_element[i]:
                good_characters += 1
                local_good += 1
            if str(text[i].upper()) == stripped_element[i].upper():
                good_characters_case_insensitive += 1
        clear_directory()
    file.write(
        f'Out of possible {all_characters} characters \n'
        f'Emnist_Deep model recognized correctly {good_characters}, which turns out to be {(good_characters / all_characters) * 100} percent accuracy\n'
        f'Emnist_Deep but when case sensitivity is not important, then correct recognition jumps up to {good_characters_case_insensitive} and that is {(good_characters_case_insensitive/ all_characters)*100} percent accuracy\n'
        f'TesseractOCR was able to recognize {tesseract_predictions} characters correctly, which is {(tesseract_predictions/all_characters)*100} percent accuracry\n'
        f'EasyOCR was able to recognize {easy_predictions} characters correctly, which is {(easy_predictions/all_characters)*100} percent accuracry\n'
        f'PaddleOCR was able to recognize {paddle_predictions} characters correctly, which is {(paddle_predictions/all_characters)*100} percent accuracry')
