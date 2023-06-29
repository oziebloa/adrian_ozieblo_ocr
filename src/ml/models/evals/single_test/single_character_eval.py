from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model

images = {'3.png': '3', '7.png': '7', '9.png': '9', 'A.png': 'A', 'B.png': 'B', 'J.png': 'J', 'K.png': 'K',
          'L.png': 'L', 'N.png': 'N', 'Z.png': 'Z'}
models = {'cnn_sequentional.h5', 'cnn_functional.h5', 'crnn.h5', 'resnet.h5'}


def label_to_char(label):
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    else:
        raise ValueError(f"Invalid label {label}. Expected a value between 0 and 35.")


def recognize_text(image_path, model_path):
    model = load_model(model_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11,
                                            4)
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmented_characters = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            character_image = threshold_image[y:y + h, x:x + w]
            resized_image = cv2.resize(character_image, (28, 28))
            segmented_characters.append(resized_image)

    recognized_text = ""

    for character_image in segmented_characters:
        normalized_image = character_image / 255.0
        reshaped_image = normalized_image.reshape(1, 28, 28, 1)
        predictions = model.predict(reshaped_image)
        predicted_label = np.argmax(predictions)
        if predicted_label < 10:
            recognized_text += str(predicted_label)
        else:
            recognized_text += chr(predicted_label + 55)

    return recognized_text


for model in models:
    model_path = Path(__file__).parent.parent.parent.parent / f'trained_models/{model}'
    good_predictions = 0
    with open('single_text_results.txt', 'a') as file:
        file.write(f'Evaluating model {model}\n')
        for img_path, img in images.items():
            prediction = recognize_text(img_path, model_path)
            file.write(f'Image containing {img} was recognized as {prediction}\n')
            if img == prediction:
                good_predictions += 1
        file.write(f'{model} was able to properly recognize {good_predictions} images \n\n\n\n')

print(Path(__file__).parent.parent.parent.parent / 'trained_models')
