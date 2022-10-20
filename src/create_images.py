from PIL import Image, ImageDraw
import random
import os

from src.CONSTS import *


def draw_circle(output_path, fill=False):
    """
    Draw a single circle and save it to the specified file
    """

    img_size = 600

    back_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )

    image = Image.new("RGB", (img_size, img_size), back_color)
    draw = ImageDraw.Draw(image)

    size_dif = random.randint(100, 400)

    x0 = random.randint(0, img_size - size_dif)
    y0 = random.randint(0, img_size - size_dif)

    x1 = x0 + size_dif
    y1 = y0 + size_dif

    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if fill:
        draw.ellipse((x0, y0, x1, y1), fill=color, width=random.randint(1, 4))
    else:
        draw.arc(
            (x0, y0, x1, y1), start=0, end=360, fill=color, width=random.randint(1, 4)
        )
    image.save(output_path)


def draw_circles():
    """ "
    Draw 25 filled and 25 normal circles and save them to their respective paths.
    """
    for i in range(25):
        draw_circle(CIRCLES_FILLED_IMAGES_PATH + "arc" + str(i) + ".jpg", fill=True)
    for i in range(25):
        draw_circle(CIRCLES_IMAGES_PATH + "/arc" + str(i) + ".jpg", fill=False)


def draw_and_save_copy(imgpath, savepath, text, name):
    """
    Write text on specific image.
    """
    image = Image.open(imgpath)
    copy = image.copy()
    draw_image = ImageDraw.Draw(copy)
    draw_image.text((50, 50), text, fill="white")
    copy.save(os.path.abspath(savepath) + "/" + name + ".jpg")


def generate_image_with_texts(CLASS_PATH=BALLS_PATH):
    """
    Generate images with texts
    """
    os.mkdir(TEXT_IMAGES_PATH)
    for typefolder in os.listdir(CLASS_PATH):
        os.mkdir(TEXT_IMAGES_PATH + "/" + typefolder)
        for class_ in os.listdir(CLASS_PATH + "/" + typefolder):
            os.mkdir(TEXT_IMAGES_PATH + "/" + typefolder + "/" + class_)
            for i, img in enumerate(
                os.listdir(CLASS_PATH + "/" + typefolder + "/" + class_)
            ):
                draw_and_save_copy(
                    imgpath="."
                    + CLASS_PATH
                    + "/"
                    + typefolder
                    + "/"
                    + class_
                    + "/"
                    + img,
                    savepath="."
                    + DATA_PATH
                    + "/textimages/"
                    + typefolder
                    + "/"
                    + class_,
                    text=class_,
                    name=str(i),
                )
