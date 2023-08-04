import random
import string
from PIL import Image, ImageDraw, ImageFont

def generate_random_string(length):
    """Generate a random string of specified length"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def generate_image_with_characters(image_width, image_height, spacing=2):
    """Generate an image with random characters and save the character string to a text file"""
    image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))  # Set white background
    draw = ImageDraw.Draw(image)
    characters = generate_random_string(36)
    text_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
    text_width, text_height = draw.textsize(characters, font=text_font)
    text_position = ((image_width - text_width) // 2, (image_height - text_height) // 2)

    character_spacing = spacing
    current_x = text_position[0]
    for char in characters:
        draw.text((current_x, text_position[1]), char, font=text_font, fill=(0, 0, 0))
        char_width, _ = draw.textsize(char, font=text_font)
        current_x += char_width + character_spacing

    image.save(f"images/{characters}.png")

    return characters


with open('labels.txt', 'a') as file:
    for i in range(500):
        label = generate_image_with_characters(800, 200)
        file.write(f'{label} \n')
