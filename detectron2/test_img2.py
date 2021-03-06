# https://rosenfelder.ai/Instance_Image_Segmentation_for_Window_and_Building_Detection_with_detectron2/
# -*- coding: utf-8 -*-

# from detectron2.utils.logger import setup_logger
# setup_logger()
# import cv2
# import os
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog


import pickle
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
import pandas as pd




# define fonts and colors
# font_id = ImageFont.truetype("/content/Roboto-Regular.ttf", 15)
# font_result = ImageFont.truetype("/content/Roboto-Regular.ttf", 40)
text_color = (255, 255, 255, 128)
background_bbox_window = (0, 247, 255, 30)
background_bbox_building = (255, 167, 14, 30)
background_text = (0, 0, 0, 150)
background_mask_window = (0, 247, 255, 100)
background_mask_building = (255, 167, 14, 100)
device = "cpu"

# this variable is True if you want to plot the output images, False if you only need
# the CSV
plot_data = True


def draw_bounding_box(img, bounding_box, text, category, id, draw_box=False):
    """Draws a bounding box onto the image as well as the building ID and the window
    percentage."""

    x = bounding_box[0]
    y = bounding_box[3]
    text = str(round(text, 2))
    draw = ImageDraw.Draw(img, "RGBA")
    # draw_box will draw the bounding box as seen in the outputs of detectron2
    if draw_box:
        if category == 0:
            draw.rectangle(bounding_box, fill=background_bbox_window, outline=(0, 0, 0))
        elif category == 1:
            draw.rectangle(
                bounding_box, fill=background_bbox_building, outline=(0, 0, 0)
            )
    w, h = font_id.getsize(id)

    draw.rectangle((x, y, x + w, y - h), fill=background_text)
    draw.text((x, y - h), id, fill=text_color, font=font_id)
    # for buildings, add the window percentage value in the lower right corner
    if category == 1:
        w, h = font_result.getsize(text)
        draw.rectangle((x, y, x + w, y + h), fill=background_text)
        draw.text((x, y), text, fill=text_color, font=font_result)


def draw_mask(img, mask, category):
    """Draws a mask onto the image."""

    img = img.convert("RGBA")

    mask_RGBA = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    if category == 0:
        mask_RGBA[mask] = background_mask_window
    elif category == 1:
        mask_RGBA[mask] = background_mask_building
    mask_RGBA[~mask] = [0, 0, 0, 0]
    rgb_image = Image.fromarray(mask_RGBA).convert("RGBA")

    img = Image.alpha_composite(img, rgb_image)

    return img


def calculate_window_perc(dataset):
    """Takes a list of prediction dictionaries as input and calculates the percentage of
    window to fassade for each building. The result is save to the dataset. For building
    data, the actual percentage is saved, for windows, 1.0 is put in."""
    with open("/content/val/via_region_data.json") as f:
        json_file = json.load(f)
    for i, data in enumerate(dataset):
        data["window_percentage"] = 0
        data["pixel_area"] = 0
        data["tagged_id"] = 0
        # loop through building
        if data["category"] == 1:
            data = get_tagged_id(data, json_file)
            window_areas = []
            building_mask = data["mask"]
            building_area = np.sum(data["mask"])

            for window in dataset:
                # for each building, loop through each window
                if window["category"] == 0:
                    window["window_percentage"] = 1
                    pixels_overlapping = np.sum(window["mask"][building_mask])

                    window_areas.append(pixels_overlapping)

            window_percentage = sum(window_areas) / building_area

            data["window_percentage"] = window_percentage
            data["pixel_area"] = building_area

    return dataset



def get_tagged_id(building, json_file):
    """Searches through the via_export_json.json of the images used for inferencing and
    adds all tagged_ids to the dataset."""

    building["tagged_id"] = 0
    building["tagged_id_coords"] = 0
    # loop through the JSON annotations file
    for idx, v in enumerate(json_file.values()):
        annos = v["regions"]

        if v["filename"] == building["file_name"]:
            try:
                for annotation in annos:
                    anno = annotation["shape_attributes"]
                    # if the annotation is not a point, go to the next annotation
                    if anno["name"] != "point":
                        continue

                    if anno["name"] == "point":
                        tagged_id = annotation["region_attributes"]["tagged_id"]
                        px = anno["cx"]
                        py = anno["cy"]
                        point = [py, px]
                        # if the point location matches with the building mask, add the
                        # id to the building data
                        if building["mask"][py][px]:
                            building["tagged_id"] = tagged_id
                            building["tagged_id_coords"] = point

            except KeyError as e:
                print("Error:", e)
                return building

    return building


def create_csv(dataset):
    """Takes a list of lists of data dictionaries as input, flattens this list, creates
    a DataFrame, removes unnecessary information and saves it as a CSV."""

    # flatten list
    dataset = [item for sublist in dataset for item in sublist]
    df = pd.DataFrame(dataset)


    # calculate the percentage of the building compared to the total image size
    df["total_image_area"] = df["image_height"] * df["image_width"]
    df["building_area_perc_of_image"] = df["pixel_area"] / df["total_image_area"]
    # keep only specific columns
    df = df[
        [
            "file_name",
            "id",
            "tagged_id",
            "tagged_id_coords",
            "category",
            "pixel_area",
            "building_area_perc_of_image",
            "window_percentage",
        ]
    ]
    # only keep building information
    df = df[df["category"] == 1]


    df.to_csv("/content/result.csv")
    return(df)



def process_data(file_path, plot_data=plot_data):
    """Takes an prediction result in form of a .pkl-file and draws the mask and bounding
    box information. From these, the percentage of windows to fassade for each building
    is calculated and plotted onto the image if plot_data=True."""
    with open(file_path, "rb") as f:
        # the following lines of code extract specific data from the prediction-dict
        prediction = pickle.load(f)

        image_height = prediction["prediction"]["instances"].image_size[0]
        image_width = prediction["prediction"]["instances"].image_size[1]

        # the data is still saved on the GPU and needs to be moved to the CPU
        boxes = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_boxes"]
            .tensor.to(device)
            .numpy()
        )

        img = Image.open(prediction["file_location"])
        categories = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_classes"]
            .to(device)
            .numpy()
        )
        masks = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_masks"]
            .to(device)
            .numpy()
        )

        dataset = []
        counter_window = 0
        counter_building = 0
        # create a new data-dict as well as IDs for each building and window within an
        # image
        for i, box in enumerate(boxes):
            data = {}
            data["file_name"] = prediction["file_name"]
            data["file_location"] = prediction["file_location"]
            data["image_height"] = image_height
            data["image_width"] = image_width
            # category 0 is always a window
            if categories[i] == 0:
                data["id"] = f"w_{counter_window}"
                counter_window = counter_window + 1
            # category 1 is always a building
            elif categories[i] == 1:
                data["id"] = f"b_{counter_building}"
                counter_building = counter_building + 1

            data["bounding_box"] = box
            data["category"] = categories[i]
            data["mask"] = masks[i]
            dataset.append(data)

        dataset = calculate_window_perc(dataset)

        if plot_data:
            for i, data in enumerate(dataset):
                draw_bounding_box(
                    img,
                    data["bounding_box"],
                    data["window_percentage"],
                    data["category"],
                    data["id"],
                    draw_box=True,
                )
            for i, data in enumerate(dataset):
                img = draw_mask(img, data["mask"], data["category"])
            try:
              img.save(
                  f"/content/predictions/{data['file_name']}_prediction.png",
                  quality=95,
              )
            except UnboundLocalError as e:
              print("no annotations found, skipping")
    return dataset


def apply_mp_progress(func, n_processes, prediction_list):
    """Applies multiprocessing to a list of data. Currently does not work well in Google
    Collab."""

    p = mp.Pool(n_processes)

    res_list = []
    with tqdm(total=len(prediction_list)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, prediction_list))):
            pbar.update()
            res_list.append(res)
        pbar.close()
    p.close()
    p.join()
    return res_list

prediction_folder = Path("/home/shmuelgr/PycharmProjects/detectron2/output/14_3_3x10^4_loop")
prediction_list = []
start = datetime.now()
for i, file in enumerate(prediction_folder.glob("*.pth")):
    file = str(file)
    prediction_list.append(file)


# this is for processing on a single CPU in Colab
dataset = []

for file_location in tqdm(prediction_list):
  dataset_part = process_data(file_location)
  dataset.append(dataset_part)

# If you use this code on a local machine, comment out the four lines above and uncomment
# the line below
# dataset = apply_mp_progress(process_data, mp.cpu_count(), prediction_list)


df = create_csv(dataset)

print(datetime.now() - start)

