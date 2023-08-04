import nltk
nltk.download('words')

import random
import textwrap
from PIL import Image, ImageDraw, ImageFont
from nltk.corpus import words

def generate_random_paragraph(word_count):
    """Generate a random paragraph of specified word count"""
    english_words = words.words()
    paragraph = ' '.join(random.choice(english_words) for _ in range(word_count))
    return paragraph

def generate_image_with_paragraph(image_width, image_height, word_count, line_spacing=10):
    """Generate an image with a random paragraph and save the paragraph to a text file"""
    image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    paragraph = generate_random_paragraph(word_count)
    words = paragraph.split(' ')
    text_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
    margin = 2.5
    current_width = margin
    current_height = margin
    space_width, _ = draw.textsize(' ', font=text_font)

    for word in words:
        word_width, word_height = draw.textsize(word, font=text_font)
        if current_width + word_width + space_width > image_width - margin:
            current_width = margin
            current_height += word_height + line_spacing
        draw.text((current_width, current_height), word, font=text_font, fill=(0, 0, 0))
        current_width += word_width + space_width

    image.save(f"images/{paragraph}.png")
    return paragraph


with open('labels.txt', 'w') as file:
    for i in range(10):
        label = generate_image_with_paragraph(800, 200, random.randint(5, 12))
        file.write(f'{label} \n')
