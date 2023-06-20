import os
import time
from pathlib import Path

from gtts import gTTS

from src.ml.trained_models import easy_ocr
from src.ml.trained_models import pytesseract
from src.ml.trained_models import paddle_ocr

def get_character_recognition(file, algorithm):
    if algorithm == 'EasyOCR':
        predicted_text = easy_ocr.ocr_image(file)
    elif algorithm == 'TesseractOCR':
        predicted_text = pytesseract.ocr_image(file)
    elif algorithm == 'PaddleOCR':
        predicted_text = paddle_ocr.ocr_image(file)
    else:
        predicted_text = 'Ejror'
    print(predicted_text)
    return str(predicted_text)


def get_list_of_images_transcribed(img_list, ml_algo):
    subpath_name = f'{str(img_list[0].filename)}{str(time.time())}'
    appended_subpath = f'tmp/{subpath_name}'
    appended_subpath_download = f'tmp_download/{subpath_name}'
    path_name = Path(__file__).parent.parent.parent.absolute() / appended_subpath
    path_name_download = Path(__file__).parent.parent.parent.absolute() / appended_subpath_download
    os.makedirs(f'{path_name}')
    os.makedirs(f'{path_name_download}')
    for img in img_list:
        text = get_character_recognition(img, ml_algo)
        audio = text_to_speech(text)
        with open(f'{path_name}/{os.path.splitext(img.filename)[0]}.txt', 'w') as file:
            file.write(text)
        audio.save(f'{path_name}/{os.path.splitext(img.filename)[0]}.mp3')
        with open(f'{path_name_download}/{os.path.splitext(img.filename)[0]}.txt', 'w') as file:
            file.write(text)
        audio.save(f'{path_name_download}/{os.path.splitext(img.filename)[0]}.mp3')
    return subpath_name


def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    return tts
