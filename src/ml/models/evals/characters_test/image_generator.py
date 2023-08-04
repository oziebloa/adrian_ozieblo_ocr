from PIL import Image, ImageDraw, ImageFont
import string

def generate_image_with_character(image_width, image_height, character):
    image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    text_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 120)
    text_width, text_height = draw.textsize(str(character), font=text_font)
    text_position = ((image_width - text_width) // 2, (image_height - text_height) // 2)
    draw.text(text_position, str(character), font=text_font, fill=(0, 0, 0))
    image.save(f"character_images/{character}.png")

    return character


with open('character_labels.txt', 'w') as file:
    for i in range(10):
        label = generate_image_with_character(140, 140, i)
        file.write(f'{label}\n')

    for letter in string.ascii_lowercase:
        label = generate_image_with_character(140, 140, letter)
        file.write(f'{label}\n')

    for letter in string.ascii_uppercase:
        label = generate_image_with_character(140, 140, letter)
        file.write(f'{label}\n')
