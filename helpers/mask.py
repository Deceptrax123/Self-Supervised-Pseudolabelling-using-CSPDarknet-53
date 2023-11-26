# Script to resize boxes based on image resize factor for effective mask generation

import pandas as pd
import numpy as np
from PIL import Image
import albumentations
import cv2
from dotenv import load_dotenv
import os


def mask_image(label, boxes):
    mask_path = os.getenv("MASK")
    img = np.zeros(shape=(256, 256), dtype=np.uint8)

    for i in boxes:
        img[i[1]:i[3], i[0]:i[2]] = 255

    cv2.imwrite(mask_path+label+'.png', img)


def resize_image():
    csv_path = os.getenv("TRAIN_CSV_PATH")
    train_path = os.getenv("TRAIN_X_PATH")
    mask_path = os.getenv("MASK")

    data = pd.read_csv(csv_path)
    image_names = data['image_name']
    box_coords = data['BoxesString']

    data_dict = dict(zip(image_names, box_coords))

    for i, key in enumerate(data_dict):
        coordinates = data_dict[key]
        class_id = 1  # Wheat heads are the only objects

        if coordinates != 'no_box':
            boxes_string = coordinates.split(';')

            box_array = list()

            img = Image.open(train_path+key+'.png')
            for i in boxes_string:
                if coordinates != 'no_box':
                    coords = i.split()
                    xmin, ymin, xmax, ymax = int(coords[0]), int(
                        coords[1]), int(coords[2]), int(coords[3])

                    # 1 represents class id
                    box_array.append([xmin, ymin, xmax, ymax, class_id])

            img_np = np.array(img)
            boxes_np = np.array(box_array)

            transform = albumentations.Compose([
                albumentations.Resize(height=256, width=256, always_apply=True)
            ], bbox_params=albumentations.BboxParams(format='pascal_voc'))

            transformed = transform(image=img_np, bboxes=boxes_np)
            mask_image(label=key, boxes=np.array(
                list(map(list, transformed['bboxes']))).astype(int))

        else:
            img = np.zeros(shape=(256, 256), dtype=np.uint8)
            cv2.imwrite(mask_path+key+'.png', img)


if __name__ == '__main__':
    load_dotenv('./.env')

    resize_image()
