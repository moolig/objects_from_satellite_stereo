# https://rosenfelder.ai/Instance_Image_Segmentation_for_Window_and_Building_Detection_with_detectron2/
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import random
from datetime import datetime
import pickle
from pathlib import Path
from tqdm import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import coco
from detectron2.utils.visualizer import Visualizer


def get_building_dicts(img_dir):
    """This function loads the JSON file created with the annotator and converts it to
    the detectron2 metadata specifications.
    """
    # load the JSON file
    # json_file = os.path.join(img_dir, "via_region_data.json")
    json_file = os.path.join(img_dir, "coco_format.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through the entries in the JSON file
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        # add file_name, image_id, height and width information to the records
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]

        objs = []
        # one image can have multiple annotations, therefore this loop is needed
        for annotation in annos:
            # reformat the polygon information to fit the specifications
            anno = annotation["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            region_attributes = annotation["region_attributes"]["class"]

            # specify the category_id to match with the class.

            if "building" in region_attributes:
                category_id = 1
            elif "window" in region_attributes:
                category_id = 0

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts



# the data has to be registered within detectron2, once for the train and once for
# the val data
# for d in ["train", "val"]:
#     DatasetCatalog.register(
#         "buildings_" + d,
#         lambda d=d: get_building_dicts("/content/" + d),
#     )


model_train_name = 'airs_dataset_train'
model_val_name = 'airs_dataset_val'


root_air = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/smallAerialImageDataset//'
train_image_root = root_air + r'/train/images'
train_json_file = root_air + r'/train/coco_format.json'
val_image_root = root_air + r'/val/images'
val_json_file = root_air + r'/val/coco_format.json'
register_coco_instances(model_train_name, {}, train_json_file, train_image_root)
register_coco_instances("airs_dataset_val", {}, val_json_file, val_image_root)

building_metadata = MetadataCatalog.get(model_train_name)
# dataset_dicts = get_building_dicts(root_air)

# dataset_dicts = get_building_dicts("/content/train")
# building_metadata = MetadataCatalog.get("buildings_train")



# for i, d in enumerate(random.sample(dataset_dicts, 2)):
#     # read the image with cv2
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2_imshow(vis.get_image()[:, :, ::-1])
#     # if you want to save the files, uncomment the line below, but keep in mind that
#     # the folder inputs has to be created first
#     # plt.savefig(f"./inputs/input_{i}.jpg")




cfg = get_cfg()
# you can choose alternative models as backbone here
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
))

cfg.DATASETS.TRAIN = (model_train_name,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
# if you changed the model above, you need to adapt the following line as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR, 0.00025 seems a good start
cfg.SOLVER.MAX_ITER = (
    1000  # 1000 iterations is a good start, for better accuracy increase this value
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    512  # (default: 512), select smaller if faster training is needed
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # for the two classes window and building
cfg.OUTPUT_DIR = './output/test2'


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# !mkdir predictions
# !mkdir output_images


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
    0.70  # set the testing threshold for this model
)

# load the validation data
cfg.DATASETS.TEST = ("buildings_val",)
# create a predictor
predictor = DefaultPredictor(cfg)

start = datetime.now()

validation_folder = Path("/content/val")

dataset_dicts = coco.load_coco_json(val_json_file, val_image_root, 'airs_dataset_val')
# for d in random.sample(dataset_dicts, 3):
# for i, file in enumerate(validation_folder.glob("*.jpg")):
for i, file in enumerate(dataset_dicts):
    # this loop opens the .jpg files from the val-folder, creates a dict with the file
    # information, plots visualizations and saves the result as .pkl files.

    # file_name = file.split("/")[-1]
    file_name = file['file_name'].split("/")[-1]
    im = cv2.imread(file['file_name'])

    outputs = predictor(im)
    output_with_filename = {}
    output_with_filename["file_name"] = file_name
    output_with_filename["file_location"] = file['file_name']
    output_with_filename["prediction"] = outputs
    # the following two lines save the results as pickle objects, you could also
    # name them according to the file_name if you want to keep better track of your data
    with open(cfg.OUTPUT_DIR + f"/content/predictions/predictions_{i}.pkl", "wb") as f:
        pickle.dump(output_with_filename, f)
    v = Visualizer(
        im[:, :, ::-1],
        metadata=building_metadata,
        scale=1,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig(cfg.OUTPUT_DIR + f"/output_images/{file_name}")


print("Time needed for inferencing:", datetime.now() - start)