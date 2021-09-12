# -*- coding: utf-8 -*-

from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.data.datasets import register_coco_instances









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







# root_air = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/AIRS//'
# train_image_root = root_air + r'/train/images'
# train_json_file = root_air + r'/train/instances_shape_train2018.json'
# val_image_root = root_air + r'/val/images'
# val_json_file = root_air + r'/val/coco_format.json'
#
# register_coco_instances("airs_dataset_train", {}, train_json_file, train_image_root)
# register_coco_instances("airs_dataset_val", {}, val_json_file, val_image_root)

if __name__ == "__main__":
    img_path = r'/home/shmuelgr/PycharmProjects/polyrnn-pp-pytorch/img/aerial.jpg'
    img_path = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/AIRS/test/images/christchurch_407.tif'
    img_path = r'/home/shmuelgr/PycharmProjects/convert_dataset_type/pycococreator-master/examples/AerialInageDataset_test/val/images/austin11_w_0_h_1.jpg'
    img_path = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/smallAerialImageDataset/test/images/innsbruck15_w_4_h_8.jpg'
    # img_path = r'/media/shmuelgr/676f1b15-b0a9-4795-945e-d2a40ccdc70f/Dataset/balloon/train/16435593892_2aa8118f4a_k.jpg'


    im = cv2.imread(img_path)
    # cv2.imshow('image', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

#------------------------------------------------------------------------------------------------------------
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("airs_dataset_train",)
    cfg.DATASETS.TEST = ("airs_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
# ------------------------------------------------------------------------------------------------------------

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.WEIGHTS = r'/home/shmuelgr/PycharmProjects/detectron2/output/14_3_3x10^4_loop/model_final.pth'
    cfg.MODEL.WEIGHTS = r'/home/shmuelgr/PycharmProjects/detectron2/output/model_final.pth'


    # cfg.MODEL.WEIGHTS = "/home/shmuelgr/PycharmProjects/detectron2/output/test2/model_final.pth"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)



    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    outputs["instances"].pred_classes
    outputs["instances"].pred_boxes

    # masks = (
    #     outputs["instances"]
    #         .get_fields()["pred_masks"]
    #         .to(device)
    #         .numpy()
    # )

    crop = outputs["instances"].pred_boxes
    print(crop)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


    res_im = v.get_image()[:, :, ::-1]

    cv2.imwrite(r'/home/shmuelgr/PycharmProjects/detectron2/output/testres.tif', res_im)




    cv2.imshow('image', res_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    df = create_csv(outputs["instances"])

