import pandas as pd
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
import cv2


def blur_roi():
    # ENV Variables
    load_dotenv(".env")
    global_path = os.getenv("TRAIN_Y_PATH")
    csv_path = os.getenv("TRAIN_CSV_PATH")
    blurred_path = os.getenv("TRAIN_X_PATH")

    data = pd.read_csv(csv_path)

    image_names = data['image_name']
    box_coords = data['BoxesString']

    data_dict = dict(zip(image_names, box_coords))

    for i, key in enumerate(data_dict):
        coordinates = data_dict[key]

        if coordinates != 'no_box':
            # Gaussian blur the region of interest
            boxes_string = coordinates.split(";")

            img = cv2.imread(os.path.join(global_path, key+".png"))
            for i in boxes_string:
                coords = i.split()
                xmin, ymin, xmax, ymax = int(coords[0]), int(
                    coords[1]), int(coords[2]), int(coords[3])

                region_of_interest = img[ymin:ymax, xmin:xmax]

                blur_boxes = cv2.GaussianBlur(region_of_interest, (51, 51), 0)
                img[ymin:ymax, xmin:xmax] = blur_boxes

            cv2.imwrite(os.path.join(blurred_path, key+".png"), img)
        else:
            # Add the images without boxes to new train folder
            img = cv2.imread(os.path.join(global_path, key+".png"))
            cv2.imwrite(os.path.join(blurred_path, key+".png"), img)


def main():
    blur_roi()


if __name__ == '__main__':
    main()
