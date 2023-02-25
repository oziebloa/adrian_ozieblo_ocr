import os
import time
from pathlib import Path

from gtts import gTTS


def get_character_recognition(file, algorithm):
    return str(file)


def get_list_of_images_transcribed(img_list, ml_algo):
    subpath_name = f'{str(img_list[0].filename)}{str(time.time())}'
    appended_subpath = f'tmp/{subpath_name}'
    path_name = Path(__file__).parent.parent.parent.absolute() / appended_subpath
    os.makedirs(f'{path_name}')
    for img in img_list:
        text = get_character_recognition(f'{os.path.splitext(img.filename)[0]}', ml_algo)
        audio = text_to_speech("Test to speech test")
        with open(f'{path_name}/{os.path.splitext(img.filename)[0]}.txt', 'w') as file:
            file.write(text)
            audio.save(f'{path_name}/{os.path.splitext(img.filename)[0]}.mp3')
    return subpath_name


def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    return tts
