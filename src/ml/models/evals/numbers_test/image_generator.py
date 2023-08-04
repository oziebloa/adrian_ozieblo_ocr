from PIL import Image, ImageDraw, ImageFont

def generate_image_with_number(image_width, image_height, num):
    """Generate an image with a single number"""
    image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)


    text_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 120)
    text_width, text_height = draw.textsize(str(num), font=text_font)
    text_position = ((image_width - text_width) // 2, (image_height - text_height) // 2)
    draw.text(text_position, str(num), font=text_font, fill=(0, 0, 0))


    image.save(f"number_images/{num}.png")

    return num


with open('number_labels.txt', 'w') as file:
    for i in range(10):
        label = generate_image_with_number(140, 140, i)
        file.write(f'{label}\n')
