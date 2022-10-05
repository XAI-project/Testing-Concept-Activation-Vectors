from PIL import Image, ImageDraw
import random


def circle(output_path, fill=False):
    image = Image.new("RGB", (600, 600), "white")
    draw = ImageDraw.Draw(image)
    size_dif = random.randint(100, 300)
    x0 = random.randint(0, 200)
    x1 = x0 + size_dif
    y0 = random.randint(0, 200)
    y1 = y0 + size_dif
    color = (random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255))
    if fill:
        draw.ellipse((x0, y0, x1, y1),
                     fill=color, width=random.randint(1, 4))
    else:
        draw.arc((x0, y0, x1, y1), start=0, end=360,
                 fill=color, width=random.randint(1, 4))
    image.save(output_path)


if __name__ == "__main__":
    for i in range(25):
        circle("data/circles_filled/arc" + str(i) + ".jpg", fill=True)
    for i in range(25):
        circle("data/circles/arc" + str(i) + ".jpg", fill=False)
